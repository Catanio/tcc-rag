# TCC-RAG: Análise Comparativa de Métodos de Recuperação para RAG em Domínio Específico

![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)

Este repositório contém os códigos e experimentos desenvolvidos para o Trabalho de Conclusão de Curso (TCC) que investiga e compara diferentes métodos de recuperação de informação para uma pipeline de **Recuperação Aprimorada Generativa (RAG)**.

## 📄 Contexto do Projeto

O avanço dos Modelos de Linguagem de Grande Escala (LLMs) trouxe grandes inovações, mas também desafios, especialmente em domínios técnicos com baixo volume de dados, como documentos institucionais. Limitações como alucinações de conteúdo e falta de rastreabilidade das fontes são problemas críticos.

A abordagem RAG surge como solução, com geração textual dos LLMs com um mecanismo de recuperação que busca trechos relevantes em uma base de conhecimento confiável antes de gerar uma resposta.

Este trabalho foca em comparar diferentes abordagens para a etapa de **recuperação** dentro de um sistema RAG, projetado para responder perguntas sobre o Projeto Pedagógico do Curso (PPC) de Ciência da Computação da Universidade Federal da Fronteira Sul (UFFS).

> **Objetivo Principal:** Identificar qual estratégia de recuperação (SPLADE vs. ColBERT) se mostra mais adequada a contextos de domínio específico com vocabulário técnico e corpus reduzido, usando o tradicional BM25 e uma implementação RAG anterior como linhas de base.

## 🛠️ Metodologia

As seguintes abordagens de recuperação serão implementadas e avaliadas comparativamente:

* **SPLADE:** Um modelo que utiliza representações esparsas com expansão de vocabulário para melhorar a correspondência de termos.
* **ColBERT:** Um modelo denso baseado em interação tardia, que avalia a relevância em um nível mais fino de granularidade (token a token).
* **BM25:** Um algoritmo clássico de recuperação baseado em contagem de termos, servindo como baseline.
* **RAG com Embeddings (Implementação Anterior):** A primeira versão do sistema, que utiliza embeddings de similaridade de cosseno.

## 📂 Estrutura do Repositório

O projeto está organizado em módulos, cada um contendo uma parte da experimentação.

### `modules/retrieval/splade`
Contém a implementação da PoC (Prova de Conceito) do modelo **SPLADE**. Este módulo é responsável por carregar o modelo e transformar trechos de texto em suas representações vetoriais esparsas (Bag-of-Words ponderado).

### `modules/translation`
Este diretório guarda um experimento inicial que visava traduzir todo o corpus do português para o inglês antes de aplicar os modelos de recuperação.
* **Resultado do Experimento:** A abordagem se mostrou inviável para o objetivo final. O processo de tradução em batch com o modelo NLLB não preservava a estrutura de marcação (proveniente de arquivos Markdown), o que resultava na perda da identificação e separação dos *chunks* de texto. O código permanece no repositório como registro da experimentação.

## 🚀 Como Executar os Módulos

Para executar os módulos, navegue até o diretório raiz do projeto (`tcc-rag`) e utilize o seguinte padrão de comando:

### Módulo SPLADE

Para executar a prova de conceito do SPLADE e ver a expansão de um texto de exemplo:

```bash
python -m modules.retrieval.splade
```

### Módulo de Tradução

Para usar a ferramenta de tradução via linha de comando:

```bash
# Traduzindo uma única frase
python -m modules.translation --text "O pinto pia, a pia pinga."

# Traduzindo múltiplas frases
python -m modules.translation --text "Olá, como você está?" "Este é um teste."
```
*Lembre-se de criar um arquivo `.env` na raiz do projeto contendo seu token do Hugging Face (`HF_TOKEN=seu_token_aqui`).*

## ✅ Próximos Passos

- [ ] Implementar o retriever ColBERT.
- [ ] Desenvolver o pipeline de avaliação com as métricas propostas (precisão, cobertura, tempo de resposta).
- [ ] Gerar e validar o conjunto de dados de consultas sintéticas para os testes.
- [ ] Executar os experimentos comparativos.
- [ ] Analisar os resultados e redigir as conclusões.

## 👨‍💻 Autor

**Mauicio Catanio**

## 📜 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.