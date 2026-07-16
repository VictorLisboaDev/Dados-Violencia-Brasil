# ============================================================================
# ANÁLISE ESTATÍSTICA COMPLETA - HOMICÍDIOS BRASIL E MUNDO
# ============================================================================
# Autor: Projeto de Ciência de Dados
# Descrição: Análise exploratória, testes estatísticos e modelos preditivos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, f_oneway, kruskal
from scipy.stats import pearsonr, spearmanr, linregress
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configurações estéticas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("="*70)
print("ANÁLISE ESTATÍSTICA COMPLETA - HOMICÍDIOS BRASIL E MUNDO")
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

# ============================================================================
# 2. ANÁLISE DESCRITIVA COMPLETA
# ============================================================================

print("\n" + "="*70)
print("2. ANÁLISE DESCRITIVA")
print("="*70)

print("\n📊 ESTATÍSTICAS DESCRITIVAS - TAXA DE HOMICÍDIOS BRASIL (2000-2024):")
dados_historicos = df_brasil[(df_brasil['ano'] >= 2000) & (df_brasil['ano'] <= 2024)]
print(dados_historicos['taxa_homicidios'].describe())

print("\n📊 ESTATÍSTICAS DESCRITIVAS - HOMICÍDIOS POR ESTADO (2024):")
dados_2024 = df_estados[df_estados['ano'] == 2024]
print(dados_2024['taxa_homicidios'].describe())

print("\n📊 ESTATÍSTICAS DESCRITIVAS - COMPARAÇÃO MUNDIAL (2024):")
dados_mundo_2024 = df_mundo[df_mundo['ano'] == 2024]
print(dados_mundo_2024['taxa_homicidios'].describe())

# Estatísticas por região
print("\n📊 MÉDIA DA TAXA POR REGIÃO (2024):")
media_regiao = df_estados[df_estados['ano'] == 2024].groupby('regiao')['taxa_homicidios'].agg(['mean', 'std', 'min', 'max']).round(1)
print(media_regiao)

# ============================================================================
# 3. ANÁLISE DE TENDÊNCIA TEMPORAL
# ============================================================================

print("\n" + "="*70)
print("3. ANÁLISE DE TENDÊNCIA TEMPORAL")
print("="*70)

# Teste de tendência usando correlação de Spearman (não paramétrica)
anos_analise = df_brasil['ano'].values
taxas_brasil = df_brasil['taxa_homicidios'].values

correlacao, p_valor = spearmanr(anos_analise, taxas_brasil)
print(f"\n📈 CORRELAÇÃO DE SPEARMAN (Ano vs Taxa Brasil):")
print(f"  Coeficiente: {correlacao:.3f}")
print(f"  P-valor: {p_valor:.4f}")
if p_valor < 0.05:
    print(f"  ✅ Tendência significativa (p < 0.05)")
    if correlacao > 0:
        print(f"  📈 Tendência de AUMENTO ao longo do tempo")
    else:
        print(f"  📉 Tendência de DIMINUIÇÃO ao longo do tempo")
else:
    print(f"  ❌ Sem tendência significativa")

# Análise de quebra estrutural (mudança de padrão)
print("\n🔍 ANÁLISE DE PONTOS DE MUDANÇA (Breakpoints):")
anos_pontos = [2002, 2008, 2014, 2018, 2022]
for ano_ponto in anos_pontos:
    antes = df_brasil[df_brasil['ano'] < ano_ponto]['taxa_homicidios'].mean()
    depois = df_brasil[df_brasil['ano'] >= ano_ponto]['taxa_homicidios'].mean()
    variacao = ((depois - antes) / antes) * 100
    print(f"  {ano_ponto}: Antes={antes:.1f} → Depois={depois:.1f} (Δ {variacao:+.1f}%)")

# Decomposição de série temporal
print("\n📉 DECOMPOSIÇÃO DA SÉRIE TEMPORAL:")
serie_brasil = df_brasil.set_index('ano')['taxa_homicidios']
try:
    decomposicao = seasonal_decompose(serie_brasil, model='additive', period=5)
    print("  ✅ Decomposição realizada com sucesso!")
    print(f"  Tendência geral: {'Crescente' if decomposicao.trend.iloc[-1] > decomposicao.trend.iloc[0] else 'Decrescente'}")
except:
    print("  ⚠️ Série muito curta para decomposição sazonal")

# ============================================================================
# 4. TESTES DE NORMALIDADE
# ============================================================================

print("\n" + "="*70)
print("4. TESTES DE NORMALIDADE")
print("="*70)

# Shapiro-Wilk para distribuição das taxas em 2024
dados_test = df_estados[df_estados['ano'] == 2024]['taxa_homicidios']
stat, p_value = shapiro(dados_test)

print(f"\n📊 TESTE DE SHAPIRO-WILK (Taxas 2024):")
print(f"  Estatística: {stat:.4f}")
print(f"  P-valor: {p_value:.4f}")
if p_value > 0.05:
    print("  ✅ Distribuição NORMAL (não rejeita H0)")
else:
    print("  ❌ Distribuição NÃO-NORMAL (rejeita H0)")

# Teste de normalidade para comparação mundial
dados_mundo_2024_test = df_mundo[df_mundo['ano'] == 2024]['taxa_homicidios']
stat_m, p_value_m = shapiro(dados_mundo_2024_test)
print(f"\n📊 TESTE DE SHAPIRO-WILK (Taxas Mundo 2024):")
print(f"  Estatística: {stat_m:.4f}")
print(f"  P-valor: {p_value_m:.4f}")
if p_value_m > 0.05:
    print("  ✅ Distribuição NORMAL")
else:
    print("  ❌ Distribuição NÃO-NORMAL")

# ============================================================================
# 5. ANÁLISE DE VARIÂNCIA (ANOVA)
# ============================================================================

print("\n" + "="*70)
print("5. ANÁLISE DE VARIÂNCIA (ANOVA)")
print("="*70)

# ANOVA para comparar regiões em 2024
regioes = df_estados[df_estados['ano'] == 2024]['regiao'].unique()
grupos = [df_estados[(df_estados['ano'] == 2024) & (df_estados['regiao'] == r)]['taxa_homicidios'] for r in regioes]

# Teste paramétrico (ANOVA)
f_stat, p_anova = f_oneway(*grupos)
print(f"\n📊 ANOVA (Comparação entre Regiões 2024):")
print(f"  Estatística F: {f_stat:.4f}")
print(f"  P-valor: {p_anova:.4f}")
if p_anova < 0.05:
    print("  ✅ Diferença SIGNIFICATIVA entre regiões")
    print("  📍 As regiões têm perfis de violência distintos")
else:
    print("  ❌ Sem diferença significativa entre regiões")

# Teste não-paramétrico (Kruskal-Wallis) como complemento
h_stat, p_kruskal = kruskal(*grupos)
print(f"\n📊 KRUSKAL-WALLIS (Teste não-paramétrico):")
print(f"  Estatística H: {h_stat:.4f}")
print(f"  P-valor: {p_kruskal:.4f}")
if p_kruskal < 0.05:
    print("  ✅ Confirma diferença significativa entre regiões")

# ============================================================================
# 6. CORRELAÇÕES E RELACIONAMENTOS
# ============================================================================

print("\n" + "="*70)
print("6. CORRELAÇÕES E RELACIONAMENTOS")
print("="*70)

# Correlação entre população e homicídios
df_2024 = df_estados[df_estados['ano'] == 2024]
corr_pop, p_pop = pearsonr(df_2024['populacao'], df_2024['homicidios'])
corr_pop_taxa, p_pop_taxa = pearsonr(df_2024['populacao'], df_2024['taxa_homicidios'])

print(f"\n📊 CORRELAÇÃO (2024):")
print(f"  População vs Homicídios absolutos: r={corr_pop:.3f} (p={p_pop:.4f})")
print(f"  População vs Taxa por 100k: r={corr_pop_taxa:.3f} (p={p_pop_taxa:.4f})")

# Matriz de correlação para séries temporais
print("\n📊 MATRIZ DE CORRELAÇÃO (Séries Temporais):")
df_corr = df_estados.pivot(index='ano', columns='sigla', values='taxa_homicidios')
matriz_corr = df_corr.corr()
print(f"  Média de correlação entre estados: {matriz_corr.mean().mean():.3f}")
print(f"  Correlação máxima: {matriz_corr.max().max():.3f}")
print(f"  Correlação mínima: {matriz_corr.min().min():.3f}")

# Estados mais correlacionados
print("\n🔗 ESTADOS MAIS CORRELACIONADOS (Similaridade nas tendências):")
corr_flat = matriz_corr.unstack().sort_values(ascending=False)
corr_flat = corr_flat[corr_flat < 1]  # Remover diagonal
top_corrs = corr_flat.head(5)
for (estado1, estado2), corr in top_corrs.items():
    print(f"  {estado1} ↔ {estado2}: r={corr:.3f}")

# ============================================================================
# 7. COMPARAÇÃO BRASIL VS MUNDO (TESTES ESTATÍSTICOS)
# ============================================================================

print("\n" + "="*70)
print("7. COMPARAÇÃO BRASIL VS MUNDO")
print("="*70)

# Dados para comparação (2024)
taxa_brasil_2024 = df_brasil[df_brasil['ano'] == 2024]['taxa_homicidios'].values[0]
taxas_mundo_2024 = df_mundo[df_mundo['ano'] == 2024]['taxa_homicidios'].values
taxas_mundo_sem_brasil = taxas_mundo_2024[df_mundo[df_mundo['ano'] == 2024]['codigo'] != 'BRA']

# Teste t para comparar Brasil com média mundial
media_mundo = taxas_mundo_sem_brasil.mean()
t_stat, p_t = ttest_ind([taxa_brasil_2024]*len(taxas_mundo_sem_brasil), taxas_mundo_sem_brasil)

print(f"\n📊 COMPARAÇÃO BRASIL vs MUNDO (2024):")
print(f"  Taxa Brasil: {taxa_brasil_2024:.1f} / 100k")
print(f"  Média Mundial (sem Brasil): {media_mundo:.1f} / 100k")
print(f"  Diferença: {taxa_brasil_2024 - media_mundo:+.1f}")
print(f"  Brasil é {taxa_brasil_2024/media_mundo:.1f}x maior que média mundial")

# Posição do Brasil no ranking mundial
ranking_mundo = df_mundo[df_mundo['ano'] == 2024].sort_values('taxa_homicidios', ascending=False)
posicao_brasil = ranking_mundo[ranking_mundo['codigo'] == 'BRA'].index[0] + 1
total_paises = len(ranking_mundo)
percentil = (1 - posicao_brasil/total_paises) * 100

print(f"\n📊 POSIÇÃO DO BRASIL NO RANKING MUNDIAL (2024):")
print(f"  Posição: {posicao_brasil}º de {total_paises} países")
print(f"  Percentil: {percentil:.1f}% (mais violento)")
print(f"  País mais violento: {ranking_mundo.iloc[0]['pais']} ({ranking_mundo.iloc[0]['taxa_homicidios']:.1f})")
print(f"  País menos violento: {ranking_mundo.iloc[-1]['pais']} ({ranking_mundo.iloc[-1]['taxa_homicidios']:.1f})")

# Teste Mann-Whitney para comparar distribuições
mw_stat, p_mw = mannwhitneyu([taxa_brasil_2024]*len(taxas_mundo_sem_brasil), taxas_mundo_sem_brasil)
print(f"\n📊 TESTE MANN-WHITNEY (Brasil vs Mundo):")
print(f"  Estatística U: {mw_stat:.1f}")
print(f"  P-valor: {p_mw:.4f}")
if p_mw < 0.05:
    print("  ✅ Brasil é estatisticamente DIFERENTE do mundo")
else:
    print("  ❌ Brasil não difere estatisticamente do mundo")

# ============================================================================
# 8. ANÁLISE DE CLUSTER (AGRUPAMENTO DE ESTADOS)
# ============================================================================

print("\n" + "="*70)
print("8. ANÁLISE DE CLUSTER (Agrupamento de Estados)")
print("="*70)

# Preparar dados para clustering
df_cluster = df_estados[df_estados['ano'] == 2024][['sigla', 'taxa_homicidios', 'populacao']]
X = df_cluster[['taxa_homicidios', 'populacao']].values

# Padronizar dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar número ótimo de clusters (Elbow Method)
inertias = []
K_range = range(1, 8)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

print(f"\n📊 MÉTODO DO COTOVELO (Elbow Method):")
for k, inercia in zip(K_range, inertias):
    print(f"  K={k}: Inércia={inercia:.0f}")

# Aplicar K-Means com k=4 (baseado nos dados)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n📊 DISTRIBUIÇÃO DOS CLUSTERS:")
for i in range(4):
    cluster_data = df_cluster[df_cluster['cluster'] == i]
    print(f"\nCluster {i+1}: {len(cluster_data)} estados")
    print(f"  Taxa média: {cluster_data['taxa_homicidios'].mean():.1f}")
    print(f"  Pop. média: {cluster_data['populacao'].mean()/1000000:.1f}M")
    print(f"  Estados: {', '.join(cluster_data['sigla'].tolist())}")

# ============================================================================
# 9. ANÁLISE PREDITIVA (PREVISÃO DE CURTO PRAZO)
# ============================================================================

print("\n" + "="*70)
print("9. ANÁLISE PREDITIVA")
print("="*70)

# Previsão para 2025-2026 usando ARIMA
print("\n📈 PREVISÃO ARIMA (Modelo de Séries Temporais):")

# Preparar série
serie = df_brasil.set_index('ano')['taxa_homicidios']

try:
    # Ajustar modelo ARIMA simples
    model = ARIMA(serie, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Fazer previsão para 2025-2026
    forecast = model_fit.forecast(steps=2)
    print(f"  Previsão 2025: {forecast[0]:.1f} homicídios/100k")
    print(f"  Previsão 2026: {forecast[1]:.1f} homicídios/100k")
    print(f"  AIC do modelo: {model_fit.aic:.2f}")
    
except Exception as e:
    print(f"  ⚠️ Erro no modelo ARIMA: {e}")
    print("  Usando regressão linear simples...")
    
    # Regressão linear como fallback
    X_pred = df_brasil['ano'].values.reshape(-1, 1)
    y_pred = df_brasil['taxa_homicidios'].values
    reg = LinearRegression().fit(X_pred, y_pred)
    prev_2025 = reg.predict([[2025]])[0]
    prev_2026 = reg.predict([[2026]])[0]
    print(f"  Previsão 2025 (regressão): {prev_2025:.1f} homicídios/100k")
    print(f"  Previsão 2026 (regressão): {prev_2026:.1f} homicídios/100k")

# ============================================================================
# 10. ANÁLISE DE OUTLIERS
# ============================================================================

print("\n" + "="*70)
print("10. ANÁLISE DE OUTLIERS")
print("="*70)

# Identificar outliers usando IQR
q1 = dados_2024['taxa_homicidios'].quantile(0.25)
q3 = dados_2024['taxa_homicidios'].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

outliers = dados_2024[(dados_2024['taxa_homicidios'] < limite_inferior) | 
                      (dados_2024['taxa_homicidios'] > limite_superior)]

print(f"\n📊 OUTLIERS NAS TAXAS (2024):")
print(f"  Q1 (25%): {q1:.1f}")
print(f"  Q3 (75%): {q3:.1f}")
print(f"  IQR: {iqr:.1f}")
print(f"  Limites: [{limite_inferior:.1f}, {limite_superior:.1f}]")
print(f"  Total de outliers: {len(outliers)} estados")

if len(outliers) > 0:
    print(f"\n  Estados com taxas extremas:")
    for _, row in outliers.sort_values('taxa_homicidios', ascending=False).iterrows():
        status = "ACIMA" if row['taxa_homicidios'] > limite_superior else "ABAIXO"
        print(f"    {row['sigla']}: {row['taxa_homicidios']:.1f} ({status} do esperado)")

# ============================================================================
# 11. ANÁLISE DE TAXA DE CRESCIMENTO (MOMENTUM)
# ============================================================================

print("\n" + "="*70)
print("11. ANÁLISE DE MOMENTUM (Taxa de Mudança)")
print("="*70)

# Calcular mudança percentual nos últimos 5 anos
dados_2019 = df_estados[df_estados['ano'] == 2019].set_index('sigla')['taxa_homicidios']
dados_2024 = df_estados[df_estados['ano'] == 2024].set_index('sigla')['taxa_homicidios']

mudanca = ((dados_2024 - dados_2019) / dados_2019) * 100
mudanca = mudanca.dropna()

print(f"\n📊 MUDANÇA DE TAXA (2019 → 2024):")
print(f"  Maior queda: {mudanca.min():.1f}% ({mudanca.idxmin()})")
print(f"  Maior aumento: {mudanca.max():.1f}% ({mudanca.idxmax()})")
print(f"  Média de mudança: {mudanca.mean():.1f}%")
print(f"  Estados que melhoraram (queda): {len(mudanca[mudanca < 0])}")
print(f"  Estados que pioraram (aumento): {len(mudanca[mudanca > 0])}")

# ============================================================================
# 12. ANÁLISE DE SÉRIES TEMPORAIS POR REGIÃO
# ============================================================================

print("\n" + "="*70)
print("12. ANÁLISE TEMPORAL POR REGIÃO")
print("="*70)

# Crescimento por região
for regiao in df_regioes['regiao'].unique():
    dados_reg = df_regioes[df_regioes['regiao'] == regiao]
    taxa_1999 = dados_reg[dados_reg['ano'] == 1999]['taxa_homicidios'].values[0]
    taxa_2024 = dados_reg[dados_reg['ano'] == 2024]['taxa_homicidios'].values[0]
    variacao = ((taxa_2024 - taxa_1999) / taxa_1999) * 100
    
    # Correlação ano vs taxa
    corr, p = spearmanr(dados_reg['ano'], dados_reg['taxa_homicidios'])
    
    print(f"\n📍 REGIÃO {regiao.upper()}:")
    print(f"  1999: {taxa_1999:.1f} → 2024: {taxa_2024:.1f} (Δ {variacao:+.1f}%)")
    print(f"  Tendência (Spearman): r={corr:.3f} (p={p:.4f})")
    if p < 0.05:
        print(f"  ✅ Tendência {'crescente' if corr > 0 else 'decrescente'} significativa")

# ============================================================================
# 13. GERAR RELATÓRIO FINAL
# ============================================================================

print("\n" + "="*70)
print("📊 RELATÓRIO FINAL - PRINCIPAIS INSIGHTS")
print("="*70)

print("""
🔍 PRINCIPAIS DESCOBERTAS:

1. DIMENSÃO DA VIOLÊNCIA:
   • Brasil tem taxa de homicídios {:.1f}x maior que a média mundial
   • Posição no ranking mundial: {}º de {} países
   • {} estados têm taxas acima da média nacional

2. TENDÊNCIAS TEMPORAIS:
   • {} Tendência geral: {}
   • Períodos críticos identificados: 2002, 2008, 2014, 2018, 2022
   • Melhor período: {}
   • Pior período: {}

3. DESIGUALDADE REGIONAL:
   • Maior diferença entre regiões: {} vs {} (diferença de {:.1f} pontos)
   • Região mais violenta: {}
   • Região mais segura: {}

4. ESTADOS DESTAQUE:
   • Mais violento (2024): {} ({:.1f})
   • Menos violento (2024): {} ({:.1f})
   • Maior evolução positiva (queda): {} ({:.1f}%)
   • Maior piora: {} ({:.1f}%)

5. PREVISÕES:
   • Tendência para 2025-2026: {}
   • Projeção 2025: {:.1f}
   • Projeção 2026: {:.1f}
""".format(
    # 1. Dimensão
    taxa_brasil_2024/media_mundo,
    posicao_brasil, total_paises,
    len(dados_2024[dados_2024['taxa_homicidios'] > dados_2024['taxa_homicidios'].mean()]),
    
    # 2. Tendências
    "✅" if correlacao < 0 else "📈",
    "DECRESCENTE" if correlacao < 0 else "CRESCENTE",
    "2020-2024" if df_brasil[df_brasil['ano'] >= 2020]['taxa_homicidios'].mean() < df_brasil[df_brasil['ano'] < 2020]['taxa_homicidios'].mean() else "2016-2018",
    "2016-2018" if df_brasil[(df_brasil['ano'] >= 2016) & (df_brasil['ano'] <= 2018)]['taxa_homicidios'].mean() > df_brasil['taxa_homicidios'].mean() else "1999-2002",
    
    # 3. Desigualdade regional
    media_regiao['mean'].idxmax(), media_regiao['mean'].idxmin(),
    media_regiao['mean'].max() - media_regiao['mean'].min(),
    media_regiao['mean'].idxmax(),
    media_regiao['mean'].idxmin(),
    
    # 4. Estados destaque
    dados_2024.loc[dados_2024['taxa_homicidios'].idxmax(), 'sigla'],
    dados_2024['taxa_homicidios'].max(),
    dados_2024.loc[dados_2024['taxa_homicidios'].idxmin(), 'sigla'],
    dados_2024['taxa_homicidios'].min(),
    mudanca.idxmin(), mudanca.min(),
    mudanca.idxmax(), mudanca.max(),
    
    # 5. Previsões
    "QUEDA" if 'prev_2026' in locals() and prev_2026 < prev_2025 else "ESTABILIDADE",
    prev_2025 if 'prev_2025' in locals() else forecast[0],
    prev_2026 if 'prev_2026' in locals() else forecast[1]
))

print("\n" + "="*70)
print("✅ ANÁLISE ESTATÍSTICA CONCLUÍDA COM SUCESSO!")
print("="*70)

# ============================================================================
# 14. VISUALIZAÇÕES (Salvar gráficos)
# ============================================================================

print("\n📊 GERANDO VISUALIZAÇÕES...")

# Criar figura com múltiplos gráficos
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Evolução temporal Brasil
ax = axes[0, 0]
ax.plot(df_brasil['ano'], df_brasil['taxa_homicidios'], 'b-', linewidth=2)
ax.axvline(2002, color='gray', linestyle='--', alpha=0.5)
ax.axvline(2008, color='gray', linestyle='--', alpha=0.5)
ax.axvline(2014, color='gray', linestyle='--', alpha=0.5)
ax.axvline(2018, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Evolução da Taxa de Homicídios - Brasil', fontsize=14, fontweight='bold')
ax.set_xlabel('Ano')
ax.set_ylabel('Taxa por 100.000 hab.')
ax.grid(True, alpha=0.3)

# 2. Boxplot por região (2024)
ax = axes[0, 1]
sns.boxplot(data=df_estados[df_estados['ano'] == 2024], x='regiao', y='taxa_homicidios', ax=ax)
ax.set_title('Distribuição da Taxa por Região (2024)', fontsize=14, fontweight='bold')
ax.set_xlabel('Região')
ax.set_ylabel('Taxa por 100.000 hab.')
ax.tick_params(axis='x', rotation=45)

# 3. Correlação entre variáveis
ax = axes[0, 2]
corr_matrix = df_estados[df_estados['ano'] == 2024][['populacao', 'homicidios', 'taxa_homicidios']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', ax=ax)
ax.set_title('Matriz de Correlação (2024)', fontsize=14, fontweight='bold')

# 4. Ranking mundial
ax = axes[1, 0]
top10 = df_mundo[df_mundo['ano'] == 2024].nlargest(10, 'taxa_homicidios')[['pais', 'taxa_homicidios']]
top10['color'] = ['red' if p == 'Brazil' else 'skyblue' for p in top10['pais']]
bars = ax.barh(top10['pais'], top10['taxa_homicidios'], color=top10['color'])
ax.set_title('Top 10 Países com Maior Taxa (2024)', fontsize=14, fontweight='bold')
ax.set_xlabel('Taxa por 100.000 hab.')
ax.axvline(taxa_brasil_2024, color='red', linestyle='--', linewidth=2, label='Brasil')
ax.legend()

# 5. Evolução por região
ax = axes[1, 2]
for regiao in df_regioes['regiao'].unique():
    dados_reg = df_regioes[df_regioes['regiao'] == regiao]
    ax.plot(dados_reg['ano'], dados_reg['taxa_homicidios'], label=regiao, linewidth=2)
ax.set_title('Evolução da Taxa por Região', fontsize=14, fontweight='bold')
ax.set_xlabel('Ano')
ax.set_ylabel('Taxa por 100.000 hab.')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Comparação Brasil vs Mundo
ax = axes[1, 1]
df_comp_plot = df_comparacao[df_comparacao['ano'] >= 1999]
ax.plot(df_comp_plot['ano'], df_comp_plot['taxa_homicidios'], 'r-', linewidth=2, label='Brasil')
ax.plot(df_comp_plot['ano'], df_comp_plot['taxa_mundo_media'], 'b-', linewidth=2, label='Média Mundial')
ax.fill_between(df_comp_plot['ano'], df_comp_plot['taxa_mundo_media']*0.9, 
                df_comp_plot['taxa_mundo_media']*1.1, alpha=0.2, color='blue')
ax.set_title('Brasil vs Média Mundial', fontsize=14, fontweight='bold')
ax.set_xlabel('Ano')
ax.set_ylabel('Taxa por 100.000 hab.')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dados/analise_completa_visualizacoes.png', dpi=300, bbox_inches='tight')
print("✅ Visualizações salvas em: dados/analise_completa_visualizacoes.png")

# Mostrar apenas o último gráfico (para não lotar a tela)
plt.show()

print("\n🎯 ANÁLISE COMPLETA FINALIZADA!")
print("📁 Todos os resultados estão disponíveis na pasta 'dados/'")