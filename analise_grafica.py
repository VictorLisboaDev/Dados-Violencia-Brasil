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