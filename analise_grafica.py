# ============================================================================
# ANÁLISE GRÁFICA COMPLETA - HOMICÍDIOS BRASIL E MUNDO
# ============================================================================
# Autor: Projeto de Ciência de Dados
# Descrição: Visualizações profissionais para análise de homicídios

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configurações estéticas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 11

print("="*70)
print("ANÁLISE GRÁFICA COMPLETA - HOMICÍDIOS BRASIL E MUNDO")
print("="*70)

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================

print("\n📂 CARREGANDO DADOS...")

try:
    df_estados = pd.read_csv('dados/homicidios_brasil_estados_1999_2026.csv')
    df_regioes = pd.read_csv('dados/homicidios_brasil_regioes_1999_2026.csv')
    df_brasil = pd.read_csv('dados/homicidios_brasil_nacional_1999_2026.csv')
    df_mundo = pd.read_csv('dados/homicidios_mundo_paises_1999_2026.csv')
    df_comparacao = pd.read_csv('dados/comparacao_brasil_mundo_1999_2026.csv')
    print("✅ Dados carregados com sucesso!")
except FileNotFoundError:
    print("❌ Arquivos não encontrados! Execute primeiro o código de geração.")
    exit()

# Criar pasta para gráficos
import os
os.makedirs('graficos', exist_ok=True)

# ============================================================================
# 2. GRÁFICO 1: EVOLUÇÃO TEMPORAL BRASIL
# ============================================================================

print("\n📊 GERANDO GRÁFICO 1: Evolução Temporal Brasil...")

fig1, ax1 = plt.subplots(figsize=(16, 8))

# Linha principal
ax1.plot(df_brasil['ano'], df_brasil['taxa_homicidios'], 
         color='#e74c3c', linewidth=3, marker='o', markersize=6, 
         label='Taxa de Homicídios')

# Área de preenchimento
ax1.fill_between(df_brasil['ano'], 0, df_brasil['taxa_homicidios'], 
                 alpha=0.2, color='#e74c3c')

# Linhas verticais de períodos de governo
governos = [
    {'inicio': 1999, 'fim': 2002, 'nome': 'FHC', 'cor': 'gray'},
    {'inicio': 2003, 'fim': 2010, 'nome': 'Lula I/II', 'cor': '#3498db'},
    {'inicio': 2011, 'fim': 2014, 'nome': 'Dilma I', 'cor': '#2ecc71'},
    {'inicio': 2015, 'fim': 2016, 'nome': 'Dilma II', 'cor': '#f39c12'},
    {'inicio': 2017, 'fim': 2018, 'nome': 'Temer', 'cor': '#9b59b6'},
    {'inicio': 2019, 'fim': 2022, 'nome': 'Bolsonaro', 'cor': '#e67e22'},
    {'inicio': 2023, 'fim': 2026, 'nome': 'Lula III', 'cor': '#2ecc71'}
]

for gov in governos:
    if gov['inicio'] <= 2026:
        ax1.axvspan(gov['inicio'], min(gov['fim'], 2026), 
                    alpha=0.15, color=gov['cor'], label=gov['nome'])

# Destaques de períodos
ax1.axvline(2017, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
            label='Pico da Violência (2017)')
ax1.axvline(2020, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
            label='Início da Queda (2020)')

# Anotações
ax1.annotate('Pico: 27.8/100k', xy=(2017, 27.8), xytext=(2018, 30),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=11, color='red', fontweight='bold')

ax1.annotate('Menor taxa\nem décadas', xy=(2024, df_brasil[df_brasil['ano']==2024]['taxa_homicidios'].values[0]), 
             xytext=(2021, 16),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
             fontsize=11, color='green', fontweight='bold')

# Configurações
ax1.set_title('Evolução da Taxa de Homicídios no Brasil (1999-2026)', 
              fontsize=18, fontweight='bold', pad=20)
ax1.set_xlabel('Ano', fontsize=14)
ax1.set_ylabel('Taxa por 100.000 habitantes', fontsize=14)
ax1.legend(loc='upper right', ncol=3, fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1998, 2027)
ax1.set_ylim(0, 35)

# Adicionar linha de média
media_historica = df_brasil['taxa_homicidios'].mean()
ax1.axhline(media_historica, color='black', linestyle=':', linewidth=1.5, 
            alpha=0.5, label=f'Média Histórica: {media_historica:.1f}')

plt.tight_layout()
plt.savefig('graficos/01_evolucao_temporal_brasil.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. GRÁFICO 2: COMPARAÇÃO REGIONAL
# ============================================================================

print("\n📊 GERANDO GRÁFICO 2: Comparação Regional...")

fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, figsize=(16, 12))

# 2a: Evolução por região
for regiao in df_regioes['regiao'].unique():
    dados_reg = df_regioes[df_regioes['regiao'] == regiao]
    ax2a.plot(dados_reg['ano'], dados_reg['taxa_homicidios'], 
              linewidth=2.5, label=regiao, marker='o', markersize=4)

ax2a.set_title('Evolução da Taxa por Região', fontsize=14, fontweight='bold')
ax2a.set_xlabel('Ano')
ax2a.set_ylabel('Taxa por 100.000 hab.')
ax2a.legend(loc='best', fontsize=10)
ax2a.grid(True, alpha=0.3)

# 2b: Boxplot por região (2024)
sns.boxplot(data=df_estados[df_estados['ano'] == 2024], 
            x='regiao', y='taxa_homicidios', ax=ax2b, palette='husl')
ax2b.set_title('Distribuição da Taxa por Região (2024)', fontsize=14, fontweight='bold')
ax2b.set_xlabel('Região')
ax2b.set_ylabel('Taxa por 100.000 hab.')
ax2b.tick_params(axis='x', rotation=45)

# 2c: Barras com média por região
media_regiao_2024 = df_estados[df_estados['ano'] == 2024].groupby('regiao')['taxa_homicidios'].mean().sort_values()
cores = ['#e74c3c' if x == media_regiao_2024.max() else '#3498db' for x in media_regiao_2024]
bars = ax2c.barh(media_regiao_2024.index, media_regiao_2024.values, color=cores, alpha=0.8)
ax2c.set_title('Média da Taxa por Região (2024)', fontsize=14, fontweight='bold')
ax2c.set_xlabel('Taxa Média por 100.000 hab.')
for i, (regiao, valor) in enumerate(media_regiao_2024.items()):
    ax2c.text(valor + 0.3, i, f'{valor:.1f}', va='center', fontweight='bold')

# 2d: Heatmap da correlação entre regiões
pivot_regioes = df_estados.pivot_table(index='ano', columns='regiao', values='taxa_homicidios')
corr_regioes = pivot_regioes.corr()
sns.heatmap(corr_regioes, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=ax2d, cbar_kws={'label': 'Correlação'})
ax2d.set_title('Correlação entre Regiões (Séries Temporais)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('graficos/02_comparacao_regional.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. GRÁFICO 3: RANKING ESTADUAL
# ============================================================================

print("\n📊 GERANDO GRÁFICO 3: Ranking Estadual...")

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(18, 10))

# 3a: Top 10 estados mais violentos (2024)
top10_2024 = df_estados[df_estados['ano'] == 2024].nlargest(10, 'taxa_homicidios')
cores = plt.cm.Reds(np.linspace(0.3, 0.9, len(top10_2024)))[::-1]
bars = ax3a.barh(top10_2024['estado'], top10_2024['taxa_homicidios'], color=cores)
ax3a.set_title('Top 10 Estados Mais Violentos (2024)', fontsize=14, fontweight='bold')
ax3a.set_xlabel('Taxa por 100.000 hab.')
ax3a.set_ylabel('Estado')
for i, (_, row) in enumerate(top10_2024.iterrows()):
    ax3a.text(row['taxa_homicidios'] + 0.3, i, f"{row['taxa_homicidios']:.1f}",
              va='center', fontweight='bold', fontsize=10)
    ax3a.text(row['taxa_homicidios'] - 0.3, i, f"({row['regiao']})",
              va='center', ha='right', fontsize=8, color='gray')

# 3b: Bottom 10 estados menos violentos (2024)
bottom10_2024 = df_estados[df_estados['ano'] == 2024].nsmallest(10, 'taxa_homicidios')
cores = plt.cm.Greens(np.linspace(0.3, 0.9, len(bottom10_2024)))[::-1]
bars = ax3b.barh(bottom10_2024['estado'], bottom10_2024['taxa_homicidios'], color=cores)
ax3b.set_title('Top 10 Estados Menos Violentos (2024)', fontsize=14, fontweight='bold')
ax3b.set_xlabel('Taxa por 100.000 hab.')
ax3b.set_ylabel('Estado')
for i, (_, row) in enumerate(bottom10_2024.iterrows()):
    ax3b.text(row['taxa_homicidios'] + 0.3, i, f"{row['taxa_homicidios']:.1f}",
              va='center', fontweight='bold', fontsize=10)
    ax3b.text(row['taxa_homicidios'] - 0.3, i, f"({row['regiao']})",
              va='center', ha='right', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('graficos/03_ranking_estadual.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. GRÁFICO 4: MAPA DE CALOR - MATRIZ DE CORRELAÇÃO
# ============================================================================

print("\n📊 GERANDO GRÁFICO 4: Matriz de Correlação...")

# Preparar dados para matriz de correlação
pivot_estados = df_estados.pivot_table(index='ano', columns='sigla', values='taxa_homicidios')
corr_matrix = pivot_estados.corr()

fig4, ax4 = plt.subplots(figsize=(20, 18))

# Criar máscara para o triângulo superior
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Heatmap com máscara
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, 
            annot=False, fmt='.2f', ax=ax4,
            cbar_kws={'label': 'Correlação', 'shrink': 0.8},
            square=True, linewidths=0.5)

ax4.set_title('Matriz de Correlação entre Estados (1999-2026)', 
              fontsize=18, fontweight='bold', pad=20)
ax4.set_xlabel('Estado', fontsize=12)
ax4.set_ylabel('Estado', fontsize=12)

plt.tight_layout()
plt.savefig('graficos/04_matriz_correlacao_estados.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. GRÁFICO 5: COMPARAÇÃO BRASIL VS MUNDO
# ============================================================================

print("\n📊 GERANDO GRÁFICO 5: Comparação Brasil vs Mundo...")

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(18, 8))

# 5a: Evolução comparativa
ax5a.plot(df_comparacao['ano'], df_comparacao['taxa_homicidios'], 
          color='red', linewidth=3, label='Brasil', marker='o', markersize=5)
ax5a.plot(df_comparacao['ano'], df_comparacao['taxa_mundo_media'], 
          color='blue', linewidth=3, label='Média Mundial', marker='s', markersize=5)

# Área entre as curvas (diferença)
ax5a.fill_between(df_comparacao['ano'], 
                  df_comparacao['taxa_homicidios'], 
                  df_comparacao['taxa_mundo_media'],
                  where=(df_comparacao['taxa_homicidios'] > df_comparacao['taxa_mundo_media']),
                  color='red', alpha=0.2, label='Brasil acima da média')
ax5a.fill_between(df_comparacao['ano'], 
                  df_comparacao['taxa_homicidios'], 
                  df_comparacao['taxa_mundo_media'],
                  where=(df_comparacao['taxa_homicidios'] <= df_comparacao['taxa_mundo_media']),
                  color='blue', alpha=0.2, label='Brasil abaixo da média')

ax5a.set_title('Brasil vs Média Mundial de Homicídios', fontsize=16, fontweight='bold')
ax5a.set_xlabel('Ano', fontsize=12)
ax5a.set_ylabel('Taxa por 100.000 hab.', fontsize=12)
ax5a.legend(loc='upper right', fontsize=11)
ax5a.grid(True, alpha=0.3)

# 5b: Razão Brasil/Mundo
razao = df_comparacao['taxa_homicidios'] / df_comparacao['taxa_mundo_media']
ax5b.plot(df_comparacao['ano'], razao, color='purple', linewidth=2.5, marker='d', markersize=5)
ax5b.axhline(1, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Igualdade (1x)')
ax5b.axhline(razao.mean(), color='orange', linestyle=':', linewidth=1.5, 
             label=f'Média: {razao.mean():.1f}x')

# Preencher área
ax5b.fill_between(df_comparacao['ano'], 0, razao, alpha=0.2, color='purple')

ax5b.set_title('Razão Brasil/Média Mundial de Homicídios', fontsize=16, fontweight='bold')
ax5b.set_xlabel('Ano', fontsize=12)
ax5b.set_ylabel('Razão (Brasil / Mundo)', fontsize=12)
ax5b.legend(loc='best', fontsize=11)
ax5b.grid(True, alpha=0.3)

# Anotação do valor atual
ultimo_ano = df_comparacao['ano'].max()
ultima_razao = razao.iloc[-1]
ax5b.annotate(f'{ultima_razao:.1f}x', 
             xy=(ultimo_ano, ultima_razao),
             xytext=(ultimo_ano-1, ultima_razao+0.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('graficos/05_comparacao_brasil_mundo.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. GRÁFICO 6: RANKING MUNDIAL
# ============================================================================

print("\n📊 GERANDO GRÁFICO 6: Ranking Mundial...")

fig6, ax6 = plt.subplots(figsize=(14, 10))

# Selecionar top 20 + Brasil (se não estiver no top 20)
df_2024_mundo = df_mundo[df_mundo['ano'] == 2024].copy()
df_2024_mundo = df_2024_mundo.sort_values('taxa_homicidios', ascending=False)

# Identificar posição do Brasil
pos_br = df_2024_mundo[df_2024_mundo['codigo'] == 'BRA'].index[0]
top_20 = df_2024_mundo.head(20)

# Se Brasil não estiver no top 20, adicionar
if 'BRA' not in top_20['codigo'].values:
    top_20 = pd.concat([top_20, df_2024_mundo[df_2024_mundo['codigo'] == 'BRA']])

# Criar gráfico com cores diferentes para Brasil
cores = ['#e74c3c' if x == 'BRA' else '#3498db' for x in top_20['codigo']]
bars = ax6.barh(top_20['pais'], top_20['taxa_homicidios'], color=cores, alpha=0.8)

# Destacar Brasil
ax6.barh(top_20[top_20['codigo'] == 'BRA']['pais'], 
        top_20[top_20['codigo'] == 'BRA']['taxa_homicidios'],
        color='red', alpha=0.9, edgecolor='darkred', linewidth=3)

ax6.set_title('Top 20 Países com Maior Taxa de Homicídios (2024)', 
             fontsize=16, fontweight='bold', pad=20)
ax6.set_xlabel('Taxa por 100.000 habitantes', fontsize=13)
ax6.set_ylabel('País', fontsize=13)

# Adicionar valores
for i, (_, row) in enumerate(top_20.iterrows()):
    ax6.text(row['taxa_homicidios'] + 0.5, i, f"{row['taxa_homicidios']:.1f}",
             va='center', fontweight='bold' if row['codigo'] == 'BRA' else 'normal',
             color='darkred' if row['codigo'] == 'BRA' else 'black')
    if row['codigo'] == 'BRA':
        ax6.text(row['taxa_homicidios'] - 1, i, '⬅ BRASIL',
                 va='center', ha='right', fontweight='bold', color='red', fontsize=11)

# Linha da média mundial
media_mundo_2024 = df_2024_mundo['taxa_homicidios'].mean()
ax6.axvline(media_mundo_2024, color='green', linestyle='--', linewidth=2,
            label=f'Média Mundial: {media_mundo_2024:.1f}')

ax6.legend(loc='lower right', fontsize=11)
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('graficos/06_ranking_mundial.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. GRÁFICO 7: ANÁLISE DE TENDÊNCIAS COM REGRESSÃO
# ============================================================================

print("\n📊 GERANDO GRÁFICO 7: Análise de Tendências...")

fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(18, 8))

# 7a: Tendência com regressão linear
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df_brasil['ano'].values.reshape(-1, 1)
y = df_brasil['taxa_homicidios'].values

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

ax7a.scatter(df_brasil['ano'], df_brasil['taxa_homicidios'], 
            color='blue', alpha=0.6, s=50, label='Dados Reais')
ax7a.plot(df_brasil['ano'], y_pred, color='red', linewidth=2.5, 
         label=f'Tendência Linear (R²={r2_score(y, y_pred):.3f})')

# Previsões futuras
anos_futuro = np.arange(2025, 2027).reshape(-1, 1)
previsoes = reg.predict(anos_futuro)
ax7a.scatter([2025, 2026], previsoes, color='green', s=100, 
            marker='D', label='Previsão', zorder=5)

ax7a.set_title('Tendência e Previsão de Homicídios no Brasil', 
              fontsize=16, fontweight='bold')
ax7a.set_xlabel('Ano', fontsize=12)
ax7a.set_ylabel('Taxa por 100.000 hab.', fontsize=12)
ax7a.legend(loc='best', fontsize=11)
ax7a.grid(True, alpha=0.3)

# 7b: Taxa de mudança ano a ano
df_brasil['variacao_anual'] = df_brasil['taxa_homicidios'].pct_change() * 100

cores_var = ['green' if x < 0 else 'red' for x in df_brasil['variacao_anual'].dropna()]
ax7b.bar(df_brasil['ano'][1:], df_brasil['variacao_anual'].dropna(), 
        color=cores_var, alpha=0.7)
ax7b.axhline(0, color='black', linewidth=1)
ax7b.set_title('Variação Anual da Taxa de Homicídios (%)', 
              fontsize=16, fontweight='bold')
ax7b.set_xlabel('Ano', fontsize=12)
ax7b.set_ylabel('Variação (%)', fontsize=12)
ax7b.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('graficos/07_analise_tendencias.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 9. GRÁFICO 8: RADAR CHART - PERFIL DOS ESTADOS
# ============================================================================

print("\n📊 GERANDO GRÁFICO 8: Radar Chart...")

fig8, ax8 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Selecionar estados representativos por região
estados_selecionados = {
    'Norte': 'PA',
    'Nordeste': 'PE',
    'Centro-Oeste': 'DF',
    'Sudeste': 'SP',
    'Sul': 'SC'
}

# Preparar dados para radar
categorias = ['Taxa 2000', 'Taxa 2010', 'Taxa 2020', 'Taxa 2024', 'População (M)']
categorias_radar = [f'{c}' for c in categorias]
N = len(categorias_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Plotar cada estado
for nome, sigla in estados_selecionados.items():
    dados_estado = df_estados[df_estados['sigla'] == sigla]
    
    # Valores para o radar
    valores = [
        dados_estado[dados_estado['ano'] == 2000]['taxa_homicidios'].values[0] if 2000 in dados_estado['ano'].values else 0,
        dados_estado[dados_estado['ano'] == 2010]['taxa_homicidios'].values[0] if 2010 in dados_estado['ano'].values else 0,
        dados_estado[dados_estado['ano'] == 2020]['taxa_homicidios'].values[0] if 2020 in dados_estado['ano'].values else 0,
        dados_estado[dados_estado['ano'] == 2024]['taxa_homicidios'].values[0] if 2024 in dados_estado['ano'].values else 0,
        dados_estado[dados_estado['ano'] == 2024]['populacao'].values[0] / 1000000  # Converter para milhões
    ]
    
    # Normalizar para radar (escala 0-1)
    max_valores = [30, 30, 30, 30, 50]  # Máximos esperados
    valores_norm = [v / max_val for v, max_val in zip(valores, max_valores)]
    valores_norm += valores_norm[:1]
    
    ax8.plot(angles, valores_norm, 'o-', linewidth=2, label=nome, alpha=0.8)
    ax8.fill(angles, valores_norm, alpha=0.1)

ax8.set_xticks(angles[:-1])
ax8.set_xticklabels(categorias_radar)
ax8.set_title('Perfil Comparativo de Estados por Região\n(Dados Normalizados)', 
             fontsize=16, fontweight='bold', pad=20)
ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax8.grid(True)

plt.tight_layout()
plt.savefig('graficos/08_radar_estados.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 10. GRÁFICO 9: GRÁFICO INTERATIVO COM PLOTLY (HTML)
# ============================================================================

print("\n📊 GERANDO GRÁFICO 9: Gráfico Interativo Plotly...")

# Criar subplots interativos
fig_interativo = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Evolução Brasil e Regiões', 'Top 10 Estados (2024)',
                   'Brasil vs Mundo', 'Distribuição por Região (2024)')
)

# 1. Evolução Brasil e regiões
for regiao in df_regioes['regiao'].unique():
    dados_reg = df_regioes[df_regioes['regiao'] == regiao]
    fig_interativo.add_trace(
        go.Scatter(x=dados_reg['ano'], y=dados_reg['taxa_homicidios'],
                  mode='lines+markers', name=regiao),
        row=1, col=1
    )

# 2. Top 10 estados
top10 = df_estados[df_estados['ano'] == 2024].nlargest(10, 'taxa_homicidios')
fig_interativo.add_trace(
    go.Bar(x=top10['taxa_homicidios'], y=top10['estado'],
           orientation='h', name='Top 10', marker_color='coral'),
    row=1, col=2
)

# 3. Brasil vs Mundo
fig_interativo.add_trace(
    go.Scatter(x=df_comparacao['ano'], y=df_comparacao['taxa_homicidios'],
              mode='lines', name='Brasil', line=dict(color='red', width=3)),
    row=2, col=1
)
fig_interativo.add_trace(
    go.Scatter(x=df_comparacao['ano'], y=df_comparacao['taxa_mundo_media'],
              mode='lines', name='Média Mundial', line=dict(color='blue', width=3)),
    row=2, col=1
)

# 4. Distribuição por região (boxplot)
for regiao in df_regioes['regiao'].unique():
    dados_reg = df_regioes[df_regioes['regiao'] == regiao]
    fig_interativo.add_trace(
        go.Box(y=dados_reg['taxa_homicidios'], name=regiao),
        row=2, col=2
    )

# Atualizar layout
fig_interativo.update_layout(height=800, showlegend=True, title_text="Dashboard Interativo - Análise de Homicídios")
fig_interativo.update_xaxes(title_text="Ano", row=1, col=1)
fig_interativo.update_xaxes(title_text="Taxa por 100k", row=1, col=2)
fig_interativo.update_xaxes(title_text="Ano", row=2, col=1)
fig_interativo.update_yaxes(title_text="Taxa por 100k", row=1, col=1)
fig_interativo.update_yaxes(title_text="Estado", row=1, col=2)
fig_interativo.update_yaxes(title_text="Taxa por 100k", row=2, col=1)

# Salvar como HTML
fig_interativo.write_html('graficos/09_dashboard_interativo.html')
print("✅ Gráfico interativo salvo em: graficos/09_dashboard_interativo.html")

# ============================================================================
# 11. GRÁFICO 10: ANÁLISE DE CLUSTERS COM PCA
# ============================================================================

print("\n📊 GERANDO GRÁFICO 10: Análise de Clusters...")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Preparar dados
df_cluster = df_estados[df_estados['ano'] == 2024].copy()
features = ['taxa_homicidios', 'populacao']
X = df_cluster[features].values

# Padronizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

# PCA para visualização 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_cluster['pca1'] = X_pca[:, 0]
df_cluster['pca2'] = X_pca[:, 1]

# Criar gráfico
fig10, (ax10a, ax10b) = plt.subplots(1, 2, figsize=(16, 8))

# 10a: Scatter plot com clusters
scatter = ax10a.scatter(df_cluster['pca1'], df_cluster['pca2'], 
                        c=df_cluster['cluster'], cmap='viridis', 
                        s=200, alpha=0.7)

# Adicionar labels dos estados
for _, row in df_cluster.iterrows():
    ax10a.annotate(row['sigla'], (row['pca1'], row['pca2']), 
                  fontsize=9, fontweight='bold')

ax10a.set_title('Clusters de Estados (PCA - 2024)', fontsize=14, fontweight='bold')
ax10a.set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%})')
ax10a.set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%})')
ax10a.grid(True, alpha=0.3)

# Legenda para clusters
centers = pca.transform(kmeans.cluster_centers_)
for i, center in enumerate(centers):
    ax10a.scatter(center[0], center[1], c='red', s=300, marker='X', 
                 edgecolor='black', linewidth=2)
    ax10a.annotate(f'Cluster {i+1}', (center[0]+0.2, center[1]+0.2),
                  fontsize=10, fontweight='bold', color='red')

# 10b: Características dos clusters
cluster_summary = df_cluster.groupby('cluster').agg({
    'taxa_homicidios': 'mean',
    'populacao': 'mean'
}).round(1)

cluster_summary['populacao'] = cluster_summary['populacao'] / 1000000  # Converter para milhões
cluster_summary = cluster_summary.rename(columns={'taxa_homicidios': 'Taxa Média', 'populacao': 'Pop. Média (M)'})

# Criar tabela
table_data = cluster_summary.values
columns = cluster_summary.columns
ax10b.axis('tight')
ax10b.axis('off')
table = ax10b.table(cellText=table_data, rowLabels=cluster_summary.index, 
                   colLabels=columns, cellLoc='center', loc='center',
                   colColours=['#3498db']*len(columns))

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

ax10b.set_title('Características dos Clusters', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('graficos/10_analise_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 12. RELATÓRIO FINAL DOS GRÁFICOS
# ============================================================================

print("\n" + "="*70)
print("📊 RELATÓRIO DOS GRÁFICOS GERADOS")
print("="*70)

graficos_gerados = [
    '01_evolucao_temporal_brasil.png',
    '02_comparacao_regional.png',
    '03_ranking_estadual.png',
    '04_matriz_correlacao_estados.png',
    '05_comparacao_brasil_mundo.png',
    '06_ranking_mundial.png',
    '07_analise_tendencias.png',
    '08_radar_estados.png',
    '09_dashboard_interativo.html',
    '10_analise_clusters.png'
]

print("\n✅ GRÁFICOS GERADOS COM SUCESSO:")
for i, nome in enumerate(graficos_gerados, 1):
    print(f"  {i}. graficos/{nome}")

print("\n📂 Todos os gráficos estão na pasta 'graficos/'")
print("🌐 O dashboard interativo pode ser aberto em qualquer navegador")
print("\n" + "="*70)
print("🎯 ANÁLISE GRÁFICA COMPLETA FINALIZADA!")
print("="*70)