from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from foldbeam_c.config import Config
from foldbeam_c.graph import Graph, fresh_graph
from foldbeam_c.metrics import balanced_accuracy_for_threshold, logloss_for_proba
from foldbeam_c.predicates import build_pool, build_probes, split_gain, top_new_predicates
from foldbeam_c.semantic import affinity_key
from foldbeam_c.signatures import c_score, incompatible, make_signature


@dataclass
class SearchResult:
    graph: Graph
    val_score: float
    budget: int
    graph_evals: int
    lineage: list[dict[str, Any]]
    wall_seconds: float
    laplace_alpha: float = 1.0
    threshold: float = 0.5
    refine_evals: int = 0
    refine_moves: int = 0


def _score_graph(graph: Graph, X: np.ndarray, y: np.ndarray, metric: str, threshold: float = 0.5) -> float:
    proba = graph.predict_proba(X)
    if metric == "logloss":
        return logloss_for_proba(y, proba)
    if metric == "balanced_accuracy":
        return 1.0 - balanced_accuracy_for_threshold(y, proba, threshold)
    if metric == "accuracy":
        pred = (proba >= float(threshold)).astype(np.int64)
        return 1.0 - float(np.mean(pred == y))
    raise ValueError(f"Unknown metric: {metric}")


def _select_threshold_for_graph(graph: Graph, X_val: np.ndarray, y_val: np.ndarray, grid: list[float]) -> tuple[float, float]:
    if not grid:
        return 0.5, float(balanced_accuracy_for_threshold(y_val, graph.predict_proba(X_val), 0.5))
    proba = graph.predict_proba(X_val)
    best_t = float(grid[0])
    best_score = -1.0
    for t in grid:
        s = float(balanced_accuracy_for_threshold(y_val, proba, float(t)))
        if s > best_score:
            best_score = s
            best_t = float(t)
    return best_t, best_score


def search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
    factor_map: dict[str, Any] | None = None,
    *,
    budget: int,
    allow_semantic: bool = True,
    allow_reuse: bool = True,
    feature_names: list[str] | None = None,
) -> SearchResult:
    started = time.perf_counter()
    graph_cfg = config.section("graph")
    factor_map = factor_map or {"factors": [], "features": {}}
    search_metric = str(graph_cfg.get("search_metric", "logloss"))
    gain_metric = str(graph_cfg.get("gain_metric", "gini"))
    alpha = float(graph_cfg["laplace_alpha"])
    pool = build_pool(X_train, list(graph_cfg["quantiles"]))
    probes = build_probes(X_train, y_train, pool, int(graph_cfg["n_probes"]), gain_metric)
    beam: list[Graph] = [fresh_graph()]
    champion: tuple[float, Graph] | None = None
    lineage: list[dict[str, Any]] = []
    graph_evals = 0

    for round_id in range(int(graph_cfg["max_rounds"])):
        candidates: list[tuple[float, Graph, str]] = []
        for graph_index, graph in enumerate(beam):
            if graph.is_complete():
                fitted = _fitted(graph, X_train, y_train, graph_cfg)
                score = _score_graph(fitted, X_val, y_val, search_metric)
                graph_evals += 1
                candidates.append((score, fitted, "complete"))
                champion = _best(champion, score, fitted)
                continue
            edge = _frontier_edge(graph, X_train)
            edge_rows = graph.rows_for_edge(X_train, edge)
            alternatives: list[tuple[str, Graph, dict[str, Any]]] = []

            stopped = graph.clone()
            stopped.stop(edge)
            alternatives.append(("STOP", stopped, {}))
            for predicate, gain in top_new_predicates(
                X_train, y_train, edge_rows, pool, int(graph_cfg["m_new"]), int(graph_cfg["min_new_parent"]), int(graph_cfg["min_child_samples"]), gain_metric
            ):
                if graph.reachable_count() + 2 <= int(budget):
                    new_graph = graph.clone()
                    new_graph.add_new(edge, predicate)
                    alternatives.append(("NEW", new_graph, {"predicate": predicate.predicate_id, "gain": gain}))

            reuse_alts = []
            if allow_reuse:
                reuse_alts = _reuse_candidates(graph, edge, X_train, y_train, probes, factor_map, graph_cfg, allow_semantic, feature_names)
                for target_id, channel, score_c, score_a in reuse_alts:
                    try:
                        reuse_graph = graph.clone()
                        reuse_graph.add_reuse(edge, target_id)
                    except ValueError:
                        continue
                    alternatives.append(("REUSE", reuse_graph, {"target": target_id, "channel": channel, "C": score_c, "A_key": score_a}))

            scored_by_action: list[tuple[str, float, Graph, Graph, dict[str, Any]]] = []
            for action, child, meta in alternatives:
                search_child = child.clone()
                if search_child.reachable_count() >= int(budget):
                    search_child.close_all()
                fitted = _fitted(search_child, X_train, y_train, graph_cfg)
                score = _score_graph(fitted, X_val, y_val, search_metric)
                graph_evals += 1
                scored_by_action.append((action, score, search_child, fitted, meta))
                if fitted.is_complete():
                    champion = _best(champion, score, fitted)

            best_alt = min((score for action, score, *_ in scored_by_action if action in {"NEW", "STOP"}), default=float("inf"))
            for action, score, search_child, fitted, meta in scored_by_action:
                accepted = action != "REUSE" or score < best_alt
                lineage.append({"event": "adjudicate", "round": round_id, "beam_index": graph_index, "edge": f"n{edge[0]}/{edge[1]}", "action": action, "J": score, "best_alt_J": best_alt, "accepted": accepted, **meta})
                if accepted:
                    candidates.append((score, fitted if search_child.is_complete() else search_child, action))

        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1].reachable_count(), item[1].canonical_hash()))
        beam = [item[1] for item in candidates[: int(graph_cfg["beam_width"])]]
        lineage.append({"event": "beam", "round": round_id, "survivors": len(beam), "best_J": candidates[0][0]})
        if all(graph.is_complete() for graph in beam):
            break

    if champion is None:
        fallback = fresh_graph()
        fallback.close_all()
        fitted = _fitted(fallback, X_train, y_train, graph_cfg)
        champion = (_score_graph(fitted, X_val, y_val, search_metric), fitted)
        graph_evals += 1
    return SearchResult(graph=champion[1], val_score=float(champion[0]), budget=int(budget), graph_evals=graph_evals, lineage=lineage, wall_seconds=time.perf_counter() - started)


def select_budget(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
    factor_map: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
) -> SearchResult:
    graph_cfg = config.section("graph")
    selection_metric = str(graph_cfg.get("selection_metric", "val"))
    if selection_metric not in {"val", "train"}:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")
    final_metric = str(graph_cfg.get("final_selection_metric", "balanced_accuracy"))
    threshold_grid = [float(v) for v in graph_cfg.get("threshold_grid", [0.5])]

    results = [
        search(X_train, y_train, X_val, y_val, config, factor_map, budget=int(budget), feature_names=feature_names)
        for budget in graph_cfg["budget_grid"]
    ]

    X_sel, y_sel = (X_train, y_train) if selection_metric == "train" else (X_val, y_val)

    for result in results:
        refined, refine_lineage, refine_stats = refine_thresholds(result.graph, X_train, y_train, X_sel, y_sel, config)
        result.graph = refined
        if refine_stats.final_score is not None:
            result.val_score = float(refine_stats.final_score)
        result.graph_evals += int(refine_stats.evals)
        result.wall_seconds += float(refine_stats.elapsed)
        result.refine_evals = int(refine_stats.evals)
        result.refine_moves = int(refine_stats.moves)
        result.lineage = result.lineage + refine_lineage
        _select_laplace_alpha(result, X_train, y_train, X_sel, y_sel, config, final_metric, threshold_grid)
        _select_threshold(result, X_sel, y_sel, threshold_grid)

    if final_metric == "balanced_accuracy":
        for result in results:
            proba = result.graph.predict_proba(X_sel)
            bal = float(balanced_accuracy_for_threshold(y_sel, proba, result.threshold))
            result.val_score = 1.0 - bal
    elif final_metric == "logloss":
        for result in results:
            proba = result.graph.predict_proba(X_sel)
            result.val_score = logloss_for_proba(y_sel, proba)
    elif final_metric == "accuracy":
        for result in results:
            proba = result.graph.predict_proba(X_sel)
            pred = (proba >= result.threshold).astype(np.int64)
            result.val_score = 1.0 - float(np.mean(pred == y_sel))

    results.sort(key=lambda r: (r.val_score, r.graph.reachable_count(), r.graph.canonical_hash()))
    best = results[0]
    best.graph_evals = int(sum(r.graph_evals for r in results))
    best.wall_seconds = float(sum(r.wall_seconds for r in results))
    best.lineage = [
        {
            "event": "budget_selection",
            "chosen_budget": best.budget,
            "chosen_laplace_alpha": best.laplace_alpha,
            "chosen_threshold": best.threshold,
            "refine_moves": best.refine_moves,
            "candidates": [
                {"budget": r.budget, "val_score": r.val_score, "reachable": r.graph.reachable_count(), "laplace_alpha": r.laplace_alpha, "threshold": r.threshold, "refine_moves": r.refine_moves}
                for r in results
            ],
        }
    ] + best.lineage
    return best


def _select_laplace_alpha(result: SearchResult, X_train: np.ndarray, y_train: np.ndarray, X_sel: np.ndarray, y_sel: np.ndarray, config: Config, final_metric: str, threshold_grid: list[float]) -> None:
    graph_cfg = config.section("graph")
    grid = graph_cfg.get("laplace_alpha_grid")
    if not grid:
        result.laplace_alpha = float(graph_cfg["laplace_alpha"])
        return
    best_alpha = float(graph_cfg["laplace_alpha"])
    best_score = float(result.val_score)
    search_metric = str(graph_cfg.get("search_metric", "logloss"))
    for alpha in grid:
        fitted = result.graph.clone()
        fitted.fit_terminals(X_train, y_train, float(alpha))
        if final_metric == "balanced_accuracy":
            t, _ = _select_threshold_for_graph(fitted, X_sel, y_sel, threshold_grid)
            proba = fitted.predict_proba(X_sel)
            score = 1.0 - float(balanced_accuracy_for_threshold(y_sel, proba, t))
        elif final_metric == "logloss":
            score = logloss_for_proba(y_sel, fitted.predict_proba(X_sel))
        else:
            score = _score_graph(fitted, X_sel, y_sel, search_metric)
        if score < best_score - 1e-12:
            best_score = score
            best_alpha = float(alpha)
    fitted_best = result.graph.clone()
    fitted_best.fit_terminals(X_train, y_train, float(best_alpha))
    result.graph = fitted_best
    result.laplace_alpha = float(best_alpha)
    result.val_score = best_score


def _select_threshold(result: SearchResult, X_sel: np.ndarray, y_sel: np.ndarray, grid: list[float]) -> None:
    proba = result.graph.predict_proba(X_sel)
    best_t, _ = _select_threshold_for_graph(result.graph, X_sel, y_sel, grid)
    result.threshold = float(best_t)


@dataclass
class RefinementStats:
    evals: int = 0
    moves: int = 0
    elapsed: float = 0.0
    start_score: float | None = None
    final_score: float | None = None


def refine_thresholds(graph: Graph, X_train: np.ndarray, y_train: np.ndarray, X_sel: np.ndarray, y_sel: np.ndarray, config: Config) -> tuple[Graph, list[dict[str, Any]], RefinementStats]:
    ref_cfg = config.section("graph").get("refinement") or {}
    if not ref_cfg.get("enabled", False):
        return graph, [], RefinementStats()
    mode = str(ref_cfg.get("mode", "threshold"))
    if mode == "resplit":
        return _refine_mode_resplit(graph, X_train, y_train, X_sel, y_sel, config, ref_cfg)
    return _refine_mode_threshold(graph, X_train, y_train, X_sel, y_sel, config, ref_cfg)


def _refine_mode_threshold(graph, X_train, y_train, X_sel, y_sel, config, ref_cfg):
    stats = RefinementStats()
    max_passes = int(ref_cfg.get("max_passes", 3))
    grid = [float(q) for q in ref_cfg.get("quantiles", [])]
    min_child = int(ref_cfg.get("min_child", 5))
    alpha = float(config.section("graph")["laplace_alpha"])
    search_metric = str(config.section("graph").get("search_metric", "logloss"))
    started = time.perf_counter()
    current = graph.clone()
    current.fit_terminals(X_train, y_train, alpha)
    best_score = _score_graph(current, X_sel, y_sel, search_metric)
    stats.evals += 1
    stats.start_score = best_score
    lineage: list[dict[str, Any]] = [{"event": "refine_start", "mode": "threshold", "J": best_score, "max_passes": max_passes}]
    for pass_id in range(max_passes):
        moves_this_pass = 0
        for node_id in sorted(current.reachable_ids()):
            node = current.nodes[node_id]
            if node.kind != "internal" or node.feature is None:
                continue
            rows = current.rows_for_node(X_train, node_id)
            if rows.size < 2 * min_child:
                continue
            values = np.asarray(X_train[rows, int(node.feature)], dtype=np.float64)
            current_threshold = round(float(node.threshold), 12)
            candidates = {round(float(np.quantile(values, q)), 12) for q in grid}
            candidates.discard(current_threshold)
            if not candidates:
                continue
            node_best_threshold = float(node.threshold)
            node_best_score = best_score
            for threshold in sorted(candidates):
                left_n = int(np.count_nonzero(values <= threshold))
                if left_n < min_child or (rows.size - left_n) < min_child:
                    continue
                trial = current.clone()
                trial.nodes[node_id].threshold = threshold
                trial.fit_terminals(X_train, y_train, alpha)
                score = _score_graph(trial, X_sel, y_sel, search_metric)
                stats.evals += 1
                if score < node_best_score - 1e-12:
                    node_best_score = score
                    node_best_threshold = threshold
            if node_best_threshold != float(node.threshold):
                old_threshold = float(node.threshold)
                current.nodes[node_id].threshold = node_best_threshold
                current.fit_terminals(X_train, y_train, alpha)
                best_score = node_best_score
                moves_this_pass += 1
                lineage.append({"event": "refine_move", "mode": "threshold", "pass": pass_id, "node": node_id, "feature": int(node.feature), "old_threshold": old_threshold, "new_threshold": node_best_threshold, "J": best_score})
        if moves_this_pass == 0:
            break
    stats.moves = sum(1 for e in lineage if e["event"] == "refine_move")
    stats.elapsed = time.perf_counter() - started
    stats.final_score = best_score
    lineage.append({"event": "refine_end", "mode": "threshold", "J": best_score, "moves": stats.moves, "evals": stats.evals})
    return current, lineage, stats


def _refine_mode_resplit(graph, X_train, y_train, X_sel, y_sel, config, ref_cfg):
    from foldbeam_c.predicates import build_pool, split_gain

    stats = RefinementStats()
    max_passes = int(ref_cfg.get("max_passes", 3))
    grid = [float(q) for q in ref_cfg.get("quantiles", [])]
    min_child = int(ref_cfg.get("min_child", 5))
    candidate_k = int(ref_cfg.get("candidate_k", 8))
    alpha = float(config.section("graph")["laplace_alpha"])
    search_metric = str(config.section("graph").get("search_metric", "logloss"))
    gain_metric = str(config.section("graph").get("gain_metric", "gini"))
    pool = build_pool(X_train, grid)
    started = time.perf_counter()
    current = graph.clone()
    current.fit_terminals(X_train, y_train, alpha)
    best_score = _score_graph(current, X_sel, y_sel, search_metric)
    stats.evals += 1
    stats.start_score = best_score
    lineage: list[dict[str, Any]] = [{"event": "refine_start", "mode": "resplit", "J": best_score, "max_passes": max_passes}]
    for pass_id in range(max_passes):
        moves_this_pass = 0
        for node_id in sorted(current.reachable_ids()):
            node = current.nodes[node_id]
            if node.kind != "internal" or node.feature is None:
                continue
            rows = current.rows_for_node(X_train, node_id)
            if rows.size < 2 * min_child:
                continue
            old_feature = int(node.feature)
            old_threshold = round(float(node.threshold), 12)
            scored: list[tuple[float, int, float]] = []
            for predicate in pool:
                key = round(float(predicate.threshold), 12)
                if int(predicate.feature) == old_feature and key == old_threshold:
                    continue
                gain, left_n, right_n = split_gain(X_train, y_train, rows, predicate, gain_metric)
                if left_n < min_child or right_n < min_child:
                    continue
                if gain <= 0.0:
                    continue
                scored.append((gain, int(predicate.feature), float(predicate.threshold)))
            scored.sort(key=lambda item: (-item[0], item[1], item[2]))
            node_best = (best_score, old_feature, float(node.threshold))
            for _gain, feature, threshold in scored[:candidate_k]:
                trial = current.clone()
                trial.nodes[node_id].feature = int(feature)
                trial.nodes[node_id].threshold = float(threshold)
                trial.fit_terminals(X_train, y_train, alpha)
                score = _score_graph(trial, X_sel, y_sel, search_metric)
                stats.evals += 1
                if score < node_best[0] - 1e-12:
                    node_best = (score, int(feature), float(threshold))
            if node_best[1] != old_feature or node_best[2] != float(node.threshold):
                current.nodes[node_id].feature = int(node_best[1])
                current.nodes[node_id].threshold = float(node_best[2])
                current.fit_terminals(X_train, y_train, alpha)
                best_score = node_best[0]
                moves_this_pass += 1
                lineage.append({"event": "refine_move", "mode": "resplit", "pass": pass_id, "node": node_id, "old_feature": old_feature, "new_feature": int(node_best[1]), "old_threshold": old_threshold, "new_threshold": node_best[2], "J": best_score})
        if moves_this_pass == 0:
            break
    stats.moves = sum(1 for e in lineage if e["event"] == "refine_move")
    stats.elapsed = time.perf_counter() - started
    stats.final_score = best_score
    lineage.append({"event": "refine_end", "mode": "resplit", "J": best_score, "moves": stats.moves, "evals": stats.evals})
    return current, lineage, stats


def refit_final(graph: Graph, X: np.ndarray, y: np.ndarray, config: Config, laplace_alpha: float | None = None) -> Graph:
    if laplace_alpha is None:
        laplace_alpha = float(config.section("graph")["laplace_alpha"])
    fitted = graph.clone()
    fitted.fit_terminals(X, y, float(laplace_alpha))
    return fitted


def _frontier_edge(graph: Graph, X: np.ndarray) -> tuple[int, str]:
    edge_rows = graph.route_edges(X)
    return max(edge_rows, key=lambda edge: (len(edge_rows[edge]), -edge[0], edge[1]))


def _reuse_candidates(graph, edge, X, y, probes, factor_map, graph_cfg, allow_semantic, feature_names):
    rows_h = graph.rows_for_edge(X, edge)
    if len(rows_h) < int(graph_cfg["min_reuse_support"]):
        return []
    alpha = float(graph_cfg.get("laplace_alpha", 1.0))
    sig_h = make_signature(X, y, rows_h, probes, alpha)
    data_rank: list[tuple[int, float]] = []
    sem_rank: list[tuple[int, tuple[float, float, float]]] = []
    for target_id in graph.reachable_ids():
        if target_id == edge[0] or graph.would_create_cycle(edge[0], target_id):
            continue
        if graph.nodes[target_id].is_open:
            continue
        rows_v = graph.rows_for_node(X, target_id)
        if len(rows_v) < int(graph_cfg["min_reuse_support"]):
            continue
        sig_v = make_signature(X, y, rows_v, probes, alpha)
        blocked, _gap = incompatible(sig_h, sig_v, float(graph_cfg["tau_sep"]))
        if blocked:
            continue
        data_rank.append((target_id, c_score(sig_h, sig_v, float(graph_cfg["support_offset"]))))
        if allow_semantic:
            key = affinity_key(graph.path_conditions(edge[0], int(graph_cfg["p_max"])), graph.path_conditions(target_id, int(graph_cfg["p_max"])), factor_map, X.shape[1], feature_names)
            sem_rank.append((target_id, key))
    data_rank.sort(key=lambda item: (-item[1], item[0]))
    sem_rank.sort(key=lambda item: (-item[1][0], -item[1][1], -item[1][2], item[0]))
    data_top = [item[0] for item in data_rank[: int(graph_cfg["shortlist_data"])]]
    sem_top: list[int] = []
    for target_id, _key in sem_rank:
        if len(sem_top) >= int(graph_cfg["shortlist_semantic"]):
            break
        if target_id not in data_top:
            sem_top.append(target_id)
    chosen: list[int] = [*data_top, *sem_top]
    for target_id, _score in data_rank:
        if len(chosen) >= int(graph_cfg["shortlist_k"]):
            break
        if target_id not in chosen:
            chosen.append(target_id)
    by_data = dict(data_rank)
    by_sem = dict(sem_rank)
    out = []
    for target_id in chosen:
        in_data = target_id in data_top
        in_sem = target_id in sem_top
        channel = "both" if in_data and in_sem else "semantic" if in_sem else "data"
        out.append((target_id, channel, float(by_data.get(target_id, float("nan"))), by_sem.get(target_id, (0.0, 0.0, 0.0))))
    return out


def _fitted(graph: Graph, X: np.ndarray, y: np.ndarray, graph_cfg: dict[str, Any]) -> Graph:
    fitted = graph.clone()
    if not fitted.is_complete():
        fitted.close_all()
    fitted.fit_terminals(X, y, float(graph_cfg["laplace_alpha"]))
    return fitted


def _best(current: tuple[float, Graph] | None, score: float, graph: Graph) -> tuple[float, Graph]:
    if current is None:
        return float(score), graph.clone()
    key = (float(score), graph.reachable_count(), graph.canonical_hash())
    old = (current[0], current[1].reachable_count(), current[1].canonical_hash())
    return (float(score), graph.clone()) if key < old else current
