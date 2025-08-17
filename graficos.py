import matplotlib.pyplot as plt
import numpy as np

# Dados do benchmark
matriz = [1024, 2048, 4096, 8192, 16384]
tempo_cpu = [0.121088, 0.530868, 2.101968, 8.524190, 31.559052]
tempo_gpu_naive = [0.00011889, 0.00046660, 0.00184720, 0.00755430, 0.02821100]
tempo_gpu_tiled = [0.000106330, 0.000408110, 0.001613400, 0.006573000, 0.023512000]
speedup_naive = [1018.4, 1137.7, 1138.2, 1128.4, 1118.7]
speedup_tiled = [1138.7, 1300.6, 1302.8, 1296.8, 1342.5]

# Configuração para padrão científico Times New Roman
plt.rcParams.update({
    'font.family': 'serif',  # Times New Roman
    'font.size': 10,         # Tamanho base para legendas
    'axes.labelsize': 11,    # Labels dos eixos
    'axes.titlesize': 12,    # Título do gráfico
    'xtick.labelsize': 9,    # Números dos eixos
    'ytick.labelsize': 9,    # Números dos eixos
    'legend.fontsize': 10,   # Legenda
    'axes.linewidth': 1,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
})

# ========================================
# GRÁFICO 1: Tempo de execução vs Tamanho
# ========================================
plt.figure(figsize=(10, 6))
plt.plot(matriz, tempo_cpu, marker='o', label='CPU', linewidth=2, markersize=6)
plt.plot(matriz, tempo_gpu_naive, marker='s', label='GPU Naive', linewidth=2, markersize=6)
plt.plot(matriz, tempo_gpu_tiled, marker='^', label='GPU Tiled', linewidth=2, markersize=6)

plt.xlabel('Tamanho da matriz')
plt.ylabel('Tempo de execução (s)')
plt.title('Tempo de execução vs Tamanho da matriz')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('grafico1.png', dpi=300, bbox_inches='tight')
plt.close()

# ===============================
# GRÁFICO 2: Speedup vs Tamanho
# ===============================
plt.figure(figsize=(10, 6))
plt.plot(matriz, speedup_naive, marker='o', label='GPU Naive', linewidth=2, markersize=6)
plt.plot(matriz, speedup_tiled, marker='s', label='GPU Tiled', linewidth=2, markersize=6)

plt.xlabel('Tamanho da matriz')
plt.ylabel('Speedup (CPU / GPU)')
plt.title('Speedup vs Tamanho da matriz')
plt.xscale('log')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('grafico2.png', dpi=300, bbox_inches='tight')
plt.close()

# =====================================================
# GRÁFICO 3: Comparação GPU Naive vs Tiled (Barras)
# =====================================================
x = np.arange(len(matriz))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, tempo_gpu_naive, width, label='GPU Naive', alpha=0.8)
plt.bar(x + width/2, tempo_gpu_tiled, width, label='GPU Tiled', alpha=0.8)

plt.xticks(x, matriz)
plt.xlabel('Tamanho da matriz')
plt.ylabel('Tempo de execução (s)')
plt.title('Comparação GPU Naive vs GPU Tiled')
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('grafico3.png', dpi=300, bbox_inches='tight')
plt.close()