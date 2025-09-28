# plot_overall.py
import matplotlib
matplotlib.use("Agg")  # GUI 없이 파일 저장 전용 백엔드
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

CSV = "gift_trust_C.csv"          # 입력 CSV
HEATMAP_OUT = "trust_heatmap_overall.png"
GRAPH_OUT   = "trust_graph_overall_colored.png"

NODE_SIZE = 1800
FIG_W, FIG_H = 8, 6
CMAP = plt.cm.plasma  # 간선 색상 맵 (강도에 따라 색)

def build_matrix(df: pd.DataFrame, names):
    """(src,tgt) 평균 C_ij로 n×n 행렬 구성 (대각 0)"""
    n = len(names)
    idx = {a: i for i, a in enumerate(names)}
    M = np.zeros((n, n), dtype=float)
    grouped = df.groupby(["src", "tgt"])["C_ij"].mean()
    for (s, t), v in grouped.items():
        if s == t:
            continue
        i, j = idx[s], idx[t]
        M[i, j] = float(v)
    return M

def save_heatmap(M, names, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, label="Trust C_ij (avg)")
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_graph_colored(M, names, title, path):
    """간선 ‘색’으로 신뢰 강도 표시 (모든 간선 표시)"""
    n = len(names)
    # 원형 배치
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)

    # 간선 수집 (대각 제외, 0은 스킵)
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = float(M[i, j])
            if w > 0:
                edges.append((i, j, w))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_title(title)
    ax.axis("off")

    if edges:
        ws = np.array([w for _, _, w in edges], dtype=float)
        vmin, vmax = float(ws.min()), float(ws.max())
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # 간선 그리기 (색 = 강도, 두께는 고정)
        for (i, j, w) in edges:
            color = CMAP(norm(w))
            ax.annotate(
                "", xy=(xs[j], ys[j]), xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", lw=2.0, color=color, alpha=0.95)
            )

        # 컬러바
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Trust C_ij (avg)")
    else:
        ax.text(0.5, 0.5, "No edges (all zeros)", ha="center", va="center")

    # 노드
    ax.scatter(xs, ys, s=NODE_SIZE, zorder=3,
               edgecolors="black", linewidths=1.2, color="skyblue")
    for i, name in enumerate(names):
        ax.text(xs[i], ys[i], name, ha="center", va="center",
                fontsize=12, weight="bold", color="black")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def main():
    # 로드 & 전처리
    if not Path(CSV).exists():
        raise FileNotFoundError(f"{CSV} 파일이 없습니다.")
    df = pd.read_csv(CSV)
    df = df[df["src"] != df["tgt"]].copy()

    # 이름 목록 (정렬 고정)
    names = sorted(set(df["src"]).union(df["tgt"]))

    # 전체 평균 행렬
    M_all = build_matrix(df, names)

    # 저장
    save_heatmap(M_all, names, "Trust Matrix (overall average)", HEATMAP_OUT)
    save_graph_colored(M_all, names, "Trust Network (overall, colored edges)", GRAPH_OUT)
    print(f"✅ saved: {HEATMAP_OUT}, {GRAPH_OUT}")

if __name__ == "__main__":
    main()
