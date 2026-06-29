# -*- coding: utf-8 -*-
"""
Génère le rapport CIR R&D pour le projet LLMInsight.
Reproduit fidèlement la mise en page du rapport de référence rapport_cir_flowai_2025.docx.
"""

from docx import Document
from docx.shared import Pt, RGBColor, Cm, Emu, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.opc.constants import RELATIONSHIP_TYPE as RT
import copy

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def hex_color(hex_str):
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return RGBColor(r, g, b)


def set_cell_bg(cell, hex_str):
    """Fills a table cell background with the given hex color."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_str)
    tcPr.append(shd)


def add_title_para(doc, text, size_pt, color_hex, bold=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size_pt)
    run.font.color.rgb = hex_color(color_hex)
    return p


def add_heading1(doc, text):
    p = doc.add_heading(text, level=1)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_before = Pt(18)
    pf.space_after = Pt(6)
    for run in p.runs:
        run.font.color.rgb = hex_color('1A3A5C')
    return p


def add_heading2(doc, text):
    p = doc.add_heading(text, level=2)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_before = Pt(12)
    pf.space_after = Pt(4)
    for run in p.runs:
        run.font.color.rgb = hex_color('255F96')
    return p


def add_normal(doc, text, color_hex='1A1A1A'):
    p = doc.add_paragraph(style='Normal')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(6)
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    run.font.color.rgb = hex_color(color_hex)
    return p


def add_bullet(doc, text, bold_prefix=None):
    """
    Adds a bullet point. If bold_prefix is given (e.g. 'V1 — Label :'),
    that prefix is rendered in bold, and text is the rest.
    """
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_after = Pt(3)
    if bold_prefix:
        r1 = p.add_run(bold_prefix)
        r1.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = hex_color('1A1A1A')
        r2 = p.add_run(text)
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = hex_color('1A1A1A')
    else:
        run = p.add_run(text)
        run.font.size = Pt(10.5)
        run.font.color.rgb = hex_color('1A1A1A')
    return p


def add_section_divider(doc, text):
    """Reproduit les blocs de séparation de partie (texte centré en grands caractères)."""
    p = doc.add_paragraph(style='Normal')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(24)
    pf.space_after = Pt(24)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = hex_color('1A3A5C')
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Build a new document based on the template (to inherit styles & theme)
# ──────────────────────────────────────────────────────────────────────────────

# Open the reference document to reuse its styles
ref = Document('rapport_cir_flowai_2025.docx')

# Create a new blank document that inherits those styles by starting from template
doc = Document('rapport_cir_flowai_2025.docx')

# Remove all existing paragraphs and tables
for element in list(doc.element.body):
    doc.element.body.remove(element)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Couverture
# ──────────────────────────────────────────────────────────────────────────────

add_title_para(doc, 'RAPPORT DE RECHERCHE ET DÉVELOPPEMENT', 18, 'FFFFFF')
add_title_para(doc, "Crédit d'Impôt Recherche (CIR)", 14, 'ADD8E6')

doc.add_paragraph()  # Blank line

add_normal(doc, 'Projet : LLMInsight — Plateforme d\'observabilité et d\'évaluation des applications LLM')
add_normal(doc, 'Société : [Nom de l\'entreprise]')
add_normal(doc, 'Année de référence : 2025')

doc.add_page_break()

# ──────────────────────────────────────────────────────────────────────────────
# Introduction générale
# ──────────────────────────────────────────────────────────────────────────────

add_heading1(doc, 'Introduction générale')

add_normal(doc, (
    "Le présent rapport décrit les travaux de Recherche et Développement conduits autour de LLMInsight, "
    "une plateforme d'observabilité et d'évaluation pour les applications fondées sur des modèles de langage "
    "de grande taille (LLM — Large Language Models). À mesure que les LLM s'intègrent dans des systèmes de "
    "production à fort enjeu (assistants conversationnels, générateurs de code, pipelines RAG, systèmes de "
    "traduction automatique, outils de résumé automatique), la question de leur fiabilité, de leur exactitude "
    "factuelle et de leur sécurité devient critique. LLMInsight répond à ce besoin en interceptant de manière "
    "transparente les échanges entre les applications et les fournisseurs de LLM, en les enregistrant, en les "
    "protégeant contre les attaques par injection de prompt et la fuite de données personnelles, puis en "
    "évaluant automatiquement chaque réponse selon un ensemble de métriques configurables."
))

add_normal(doc, (
    "Ce rapport est structuré autour des cinq axes constitutifs du projet R&D :"
))

add_bullet(doc, "Contexte et état de l'art dans le domaine de l'observabilité des LLM.")
add_bullet(doc, "Objectifs R&D et incertitudes scientifiques identifiées.")
add_bullet(doc, "Démarche scientifique et technique mise en œuvre.")
add_bullet(doc, "Résultats obtenus.")
add_bullet(doc, "Contributions scientifiques et perspectives.")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION — Module LLMInsight
# ──────────────────────────────────────────────────────────────────────────────

add_section_divider(doc, 'LLMINSIGHT — PLATEFORME D\'OBSERVABILITÉ ET D\'ÉVALUATION DES LLM')

# ══════════════════════════════════════════════════════════════════════════════
# 1. Contexte et état de l'art
# ══════════════════════════════════════════════════════════════════════════════

add_heading1(doc, "1. Contexte et état de l'art")

add_heading2(doc, '1.1 Domaine scientifique et technique')

add_normal(doc, (
    "LLMInsight s'inscrit à l'intersection de plusieurs disciplines actives : l'évaluation automatique de la "
    "qualité des modèles de langage (LLM evaluation), la sécurité des systèmes IA (AI safety), la détection des "
    "hallucinations, et l'observabilité des systèmes logiciels distribués. Les LLM tels que GPT-4 ou les modèles "
    "Azure OpenAI sont désormais déployés dans des pipelines de production complexes — systèmes RAG "
    "(Retrieval-Augmented Generation), chatbots d'entreprise, générateurs de code, traducteurs automatiques — "
    "sans qu'existent de mécanismes standardisés pour surveiller en continu la qualité, la fidélité factuelle "
    "et la sécurité de leurs sorties. L'enjeu R&D est de concevoir une couche d'observabilité qui soit "
    "transparente pour l'application cliente, qui n'introduise pas de latence significative, et qui calcule "
    "automatiquement des métriques d'évaluation pertinentes selon la nature de la tâche."
))

add_heading2(doc, '1.2 Solutions existantes et leurs limites')

add_normal(doc, "Les approches actuelles d'observabilité et d'évaluation des LLM présentent des limitations structurelles :")

add_bullet(doc,
    "Frameworks d'évaluation isolés (RAGAS, DeepEval, TruLens) : ces outils permettent d'évaluer "
    "les LLM dans des environnements de test mais ne s'intègrent pas nativement dans un pipeline de "
    "production sans modification substantielle du code applicatif. Ils ne fournissent pas de couche proxy "
    "transparente ni de journalisation persistante.")

add_bullet(doc,
    "Outils de monitoring généralistes (Datadog, Grafana, OpenTelemetry) : conçus pour des systèmes "
    "distribués classiques, ils ne proposent pas de primitives spécialisées pour l'évaluation sémantique "
    "des sorties LLM (fidélité factuelle, pertinence de la réponse, rappel contextuel, score COMET).")

add_bullet(doc,
    "Plateformes LLMOps commerciales (LangSmith, Arize, Weights & Biases) : offrent des tableaux de bord "
    "de suivi mais nécessitent l'adoption d'un SDK propriétaire, ce qui crée une dépendance forte. "
    "Aucune ne propose de pipeline de sécurité intégré (détection d'injection de prompt, anonymisation PII).")

add_bullet(doc,
    "Métriques d'évaluation manuelles : l'évaluation humaine (Human-in-the-loop) reste coûteuse et non "
    "scalable en production. Les métriques automatiques de référence (BLEU, ROUGE) sont mal adaptées aux "
    "sorties conversationnelles libres des LLM modernes.")

add_normal(doc, "Ces approches présentent des limites structurelles communes :")

add_bullet(doc, "Absence de transparence : nécessitent une modification du code applicatif pour injecter la couche de monitoring.")
add_bullet(doc, "Silos fonctionnels : évaluation, sécurité et journalisation sont gérées par des outils séparés, sans cohérence.")
add_bullet(doc, "Métriques génériques non adaptées à la tâche : une métrique de traduction (COMET) ne se substitue pas à une métrique de fidélité RAG (context recall).")
add_bullet(doc, "Absence de gestion native des données sensibles : les PII (noms, adresses, mots de passe) circulent en clair dans les logs.")
add_bullet(doc, "Pas d'évaluation asynchrone non bloquante : le calcul des métriques pénalise la latence applicative.")

add_heading2(doc, '1.3 Verrous scientifiques et techniques')

add_normal(doc, "Les verrous suivants constituent le cœur de la problématique R&D de LLMInsight.")

add_bullet(doc,
    "Proxy LLM transparent compatible OpenAI : concevoir un point d'interception qui expose exactement "
    "la même interface que l'API OpenAI (POST /v1/chat/completions), afin que l'application cliente "
    "ne nécessite aucune modification de code, tout en gérant la sécurité, la journalisation et "
    "l'évaluation.",
    bold_prefix="V1 — ")

add_bullet(doc,
    "Sélection automatique des métriques pertinentes : déterminer, par raisonnement LLM, le sous-ensemble "
    "des métriques configurées par l'utilisateur qui sont réellement applicables à un échange donné "
    "(éviter de calculer des métriques RAG sur un échange ne disposant pas de contextes récupérés).",
    bold_prefix="V2 — ")

add_bullet(doc,
    "Évaluation non bloquante à latence constante : calculer des métriques computationnellement coûteuses "
    "(appels LLM pour la fidélité factuelle, embedding pour la pertinence, modèle COMET pour la traduction) "
    "de façon asynchrone, sans retarder la réponse renvoyée à l'application cliente.",
    bold_prefix="V3 — ")

add_bullet(doc,
    "Détection des hallucinations par décomposition en affirmations : segmenter une réponse LLM en "
    "affirmations factuelles atomiques et vérifier indépendamment chacune contre la connaissance du modèle "
    "ou des contextes RAG récupérés, pour produire un score de fidélité et une liste d'hallucinations.",
    bold_prefix="V4 — ")

add_bullet(doc,
    "Anonymisation et désanonymisation réversibles des PII dans le pipeline complet : détecter et remplacer "
    "les entités nommées sensibles (personnes, organisations, lieux, identifiants) dans les messages avant "
    "leur envoi au fournisseur LLM, puis ré-injecter les vraies valeurs dans la réponse finale, sans "
    "perturber la cohérence sémantique de l'échange.",
    bold_prefix="V5 — ")

add_bullet(doc,
    "Évaluation spécialisée par type de tâche : concevoir un catalogue de métriques hétérogènes couvrant "
    "des tâches distinctes (génération générique, RAG, résumé, traduction, génération de code) avec "
    "des méthodes d'implémentation fondamentalement différentes (cosine similarity, NLI, COMET neural, "
    "compilation syntaxique).",
    bold_prefix="V6 — ")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Objectifs du projet
# ══════════════════════════════════════════════════════════════════════════════

add_heading1(doc, '2. Objectifs du projet')

add_heading2(doc, '2.1 Objectifs R&D')

add_bullet(doc,
    "Développer un proxy HTTP transparent, compatible OpenAI, interceptant les échanges LLM sans "
    "modification du code applicatif client, avec gestion du cycle de vie complet (authentification "
    "par clé API de projet, validation, journalisation, évaluation, réponse).",
    bold_prefix="O1 : ")

add_bullet(doc,
    "Concevoir un pipeline de sécurité pré-envoi : détection d'injection de prompt (llm-guard, "
    "PromptInjection scanner), anonymisation PII par reconnaissance d'entités nommées (BERT-large NER, "
    "Vault de secrets), avec blocage conditionnel et désanonymisation post-réponse.",
    bold_prefix="O2 : ")

add_bullet(doc,
    "Implémenter un moteur d'évaluation asynchrone calculant automatiquement, pour chaque interaction "
    "journalisée, le sous-ensemble des métriques configurées applicables à l'échange, via une décision LLM "
    "préliminaire (decide_on_metrics).",
    bold_prefix="O3 : ")

add_bullet(doc,
    "Concevoir un catalogue de métriques couvrant six familles : métriques génériques (fidélité factuelle, "
    "pertinence de la réponse, modération, hallucinations), métriques RAG (précision du contexte, rappel "
    "du contexte, fidélité contextuelle), métriques de résumé (complétude, concision, ratio de compression), "
    "métriques de traduction (score COMET), métriques de code (compilabilité, pertinence du code), métriques "
    "heuristiques (égalité exacte, containment, regex, JSON valide).",
    bold_prefix="O4 : ")

add_bullet(doc,
    "Mettre en place un modèle de données structuré (MongoDB/Beanie) permettant la persistance des "
    "interactions (User, Project, Log, Record, Metric) et leur agrégation temporelle pour visualisation.",
    bold_prefix="O5 : ")

add_bullet(doc,
    "Développer une interface de visualisation (React, Recharts) affichant par projet : la liste des "
    "interactions avec leurs métadonnées (modèle, tokens, latence, température), les détails de chaque log "
    "et un tableau de bord de suivi des métriques dans le temps.",
    bold_prefix="O6 : ")

add_heading2(doc, '2.2 Incertitudes scientifiques')

add_bullet(doc,
    "Fiabilité de la décomposition en affirmations (claim extraction) par LLM pour des réponses longues "
    "ou ambiguës : le modèle peut produire des affirmations partiellement correctes ou agrégées, "
    "faussant les scores de fidélité.")

add_bullet(doc,
    "Calibration du score COMET (Unbabel/wmt22-comet-da) sur des domaines spécialisés (médical, juridique, "
    "technique) pour lesquels le modèle de référence n'a pas été entraîné.")

add_bullet(doc,
    "Robustesse du scanner d'injection de prompt (PromptInjection, seuil 0.5) face à des attaques "
    "sophistiquées multilangues ou codées, sans générer un taux excessif de faux positifs bloquant "
    "des requêtes légitimes.")

add_bullet(doc,
    "Précision de l'anonymisation NER (BERT-large) sur des entités atypiques, abréviations ou données "
    "structurées (numéros de carte, IBAN) qui peuvent échapper au modèle de reconnaissance classique.")

add_bullet(doc,
    "Pertinence de la décision automatique de sélection des métriques par LLM (decide_on_metrics) : "
    "le modèle peut omettre des métriques applicables ou en inclure d'inapplicables, introduisant "
    "un biais dans l'évaluation.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Démarche scientifique et technique
# ══════════════════════════════════════════════════════════════════════════════

add_heading1(doc, '3. Démarche scientifique et technique')

add_heading2(doc, '3.1 Architecture de la plateforme')

add_normal(doc, "LLMInsight s'articule autour de quatre composants principaux :")

add_bullet(doc,
    "Proxy d'interception LLM (POST /v1/chat/completions) : point d'entrée unique, compatible "
    "OpenAI, gérant authentification par Bearer token, sécurité, journalisation et évaluation asynchrone.")

add_bullet(doc,
    "Pipeline de sécurité : détection d'injection de prompt par llm-guard, anonymisation des messages "
    "par NER BERT-large, gestion du Vault de secrets pour la désanonymisation de la réponse.")

add_bullet(doc,
    "Moteur d'évaluation asynchrone : calcul non bloquant des métriques configurées, sélection "
    "automatique par LLM des métriques applicables, persistance des résultats dans le log MongoDB.")

add_bullet(doc,
    "API CRUD et interface React : gestion des utilisateurs, projets, logs et records via FastAPI/Beanie, "
    "visualisation des métriques agrégées dans le temps via Recharts.")

add_heading2(doc, '3.2 Proxy d\'interception LLM transparent')

add_normal(doc, (
    "Le proxy LLMInsight expose le endpoint POST /v1/chat/completions, syntaxiquement identique à l'API "
    "OpenAI, étendu de trois champs supplémentaires : project_id (identifiant du projet LLMInsight), "
    "provider_endpoint (URL du fournisseur LLM cible), provider_key (clé API du fournisseur). "
    "Cette conception permet à toute application utilisant le SDK OpenAI de pointer vers LLMInsight "
    "en modifiant uniquement le base_url, sans toucher à la logique applicative. Le flux de traitement "
    "est le suivant :"
))

add_bullet(doc, "Extraction des credentials et vérification de la clé API projet (Bearer token).")
add_bullet(doc, "Scan d'injection de prompt sur le dernier message utilisateur (PromptInjection, MatchType.FULL, seuil 0.5).")
add_bullet(doc, "Anonymisation PII de tous les messages via BERT-large NER et enregistrement dans le Vault.")
add_bullet(doc, "Journalisation du log dans MongoDB (état initial : prompt, historique, métadonnées).")
add_bullet(doc, "Transfert de la requête au fournisseur LLM cible (provider_endpoint + provider_key).")
add_bullet(doc, "Mesure de la latence et des tokens consommés.")
add_bullet(doc, "Désanonymisation de la réponse via le Vault (réinjection des vraies valeurs).")
add_bullet(doc, "Retour de la réponse formatée OpenAI à l'application cliente.")
add_bullet(doc, "Déclenchement asynchrone du calcul des métriques (asyncio.create_task) sans bloquer la réponse.")

add_heading2(doc, '3.3 Pipeline d\'évaluation automatique des métriques')

add_normal(doc, (
    "Le moteur d'évaluation repose sur la classe Metrics qui encapsule la requête, la réponse, "
    "les contextes RAG optionnels et l'ensemble des ressources partagées (modèle d'embedding "
    "BAAI/bge-large-en-v1.5, client OpenAI/Azure, modèle COMET). Avant calcul, une étape préliminaire "
    "décide_on_metrics interroge le LLM pour déterminer le sous-ensemble applicable. Les métriques "
    "sont ensuite calculées de façon mixte synchrone/asynchrone avec timeouts individuels."
))

add_normal(doc, "Les six familles de métriques implémentées sont :")

add_bullet(doc,
    "Métriques génériques : general_faithfulness (décomposition en claims + vérification LLM), "
    "answer_relevance (génération de questions inverses + similarité cosinus), moderation (score "
    "de sécurité 0–1 par LLM), hallucination (liste des claims non factuels identifiés).",
    bold_prefix="Famille 1 — ")

add_bullet(doc,
    "Métriques RAG : context_precision (précision@k des contextes récupérés par NLI LLM), "
    "context_recall (rappel par comparaison avec une réponse de référence LLM), "
    "contextual_faithfulness (proportion des claims supportés par les contextes récupérés).",
    bold_prefix="Famille 2 — ")

add_bullet(doc,
    "Métriques de résumé : summary_completeness (proportion des claims originaux couverts par "
    "le résumé via NLI), conciseness (proportion de texte non redondant), "
    "compression_ratio (rapport longueur entrée / longueur sortie).",
    bold_prefix="Famille 3 — ")

add_bullet(doc,
    "Métriques de traduction : comet_score via le modèle neuronal Unbabel/wmt22-comet-da, "
    "avec génération automatique d'une référence de traduction par GPT-4o si aucune n'est fournie.",
    bold_prefix="Famille 4 — ")

add_bullet(doc,
    "Métriques de code : compilable (compilation réelle du code généré dans un répertoire temporaire "
    "avec gcc/python/node/javac selon la langue détectée), code_relevance (similarité cosinus "
    "entre la requête et des questions générées par LLM pour le code produit).",
    bold_prefix="Famille 5 — ")

add_bullet(doc,
    "Métriques heuristiques : equals (correspondance exacte), contains (sous-chaîne), "
    "matches_regex (expression régulière), is_json (validité JSON).",
    bold_prefix="Famille 6 — ")

add_heading2(doc, '3.4 Sécurité et protection des données personnelles')

add_normal(doc, (
    "L'ensemble du pipeline garantit que les données personnelles ne transitent jamais en clair vers "
    "le fournisseur LLM. Le composant d'anonymisation repose sur llm-guard avec un recognizer BERT-large "
    "NER (BERT_LARGE_NER_CONF) pour détecter les entités nommées (personnes, organisations, lieux, "
    "dates, identifiants) et les remplacer par des placeholders uniques stockés dans un Vault. "
    "Après réception de la réponse du fournisseur, la désanonymisation est appliquée pour restituer "
    "les vraies valeurs à l'application cliente, tandis que la version anonymisée est seule persistée "
    "dans la base de données."
))

add_bullet(doc, "Détection des injections de prompt par modèle de classification fine-tuné (llm-guard PromptInjection).")
add_bullet(doc, "Anonymisation par NER BERT-large avec support des entités nommées personnalisées (hidden_names).")
add_bullet(doc, "Vault de secrets session-scoped garantissant l'isolation entre sessions parallèles.")
add_bullet(doc, "Journalisation uniquement des données anonymisées : les PII ne sont jamais persistées en clair.")
add_bullet(doc, "Blocage conditionnel avec journalisation de la raison du blocage, sans exposition de la réponse.")

add_heading2(doc, '3.5 Expérimentations')

add_bullet(doc, (
    "Calibration du seuil de détection d'injection de prompt (0.5) pour équilibrer la sensibilité "
    "aux attaques et le taux de faux positifs sur des requêtes légitimes contenant des instructions "
    "en langage naturel."
))

add_bullet(doc, (
    "Validation de la décomposition en affirmations (break_into_claims) sur des réponses de longueurs "
    "et styles variés : réponses factuelles courtes, paragraphes argumentés, listes structurées, "
    "en mesurant la granularité et l'indépendance des claims extraits."
))

add_bullet(doc, (
    "Mesure des timeouts individuels par métrique (40 s pour les métriques asynchrones, 20 s pour "
    "les métriques synchrones) pour garantir que l'évaluation ne dépasse pas un seuil raisonnable "
    "même en cas de latence élevée du fournisseur LLM."
))

add_bullet(doc, (
    "Tests d'intégration du score COMET (Unbabel/wmt22-comet-da, CPU, batch_size=1) sur des paires "
    "source/traduction de domaines variés, avec génération automatique de la référence par GPT-4o "
    "pour valider la cohérence du score en l'absence de traduction de référence humaine."
))

add_bullet(doc, (
    "Itérations sur le prompt système de decide_on_metrics pour améliorer la précision de la sélection "
    "automatique des métriques applicables, notamment la distinction entre contextes RAG fournis "
    "et métriques RAG activables."
))

# ══════════════════════════════════════════════════════════════════════════════
# 4. Résultats obtenus
# ══════════════════════════════════════════════════════════════════════════════

add_heading1(doc, '4. Résultats obtenus')

add_normal(doc, (
    "LLMInsight permet la surveillance continue et l'évaluation automatique des interactions LLM "
    "en production, avec les résultats suivants :"
))

add_bullet(doc, (
    "Proxy OpenAI-compatible opérationnel : interception transparente des requêtes POST /v1/chat/completions "
    "sans modification du code applicatif client, avec authentification par clé API de projet et "
    "gestion du cycle de vie complet."
))

add_bullet(doc, (
    "Pipeline de sécurité intégré : détection et blocage des injections de prompt, anonymisation PII "
    "réversible sur tous les messages du historique conversationnel, avec persistance uniquement des "
    "données anonymisées."
))

add_bullet(doc, (
    "Catalogue de 19 métriques opérationnelles réparties en six familles (générique, RAG, résumé, "
    "traduction, code, heuristique), sélectionnées automatiquement par LLM selon la nature de chaque "
    "interaction, calculées de façon non bloquante."
))

add_bullet(doc, (
    "Évaluation de la fidélité factuelle par décomposition en claims et vérification LLM, produisant "
    "un score de fidélité [0, 1] et une liste d'hallucinations identifiées pour chaque réponse."
))

add_bullet(doc, (
    "Score COMET neuronal (Unbabel/wmt22-comet-da) pour l'évaluation de la qualité de traduction, "
    "avec génération automatique d'une traduction de référence GPT-4o lorsqu'aucune référence humaine "
    "n'est disponible."
))

add_bullet(doc, (
    "Évaluation de la compilabilité du code généré par compilation réelle dans un environnement "
    "temporaire isolé, supportant Python, C, C++, Java, JavaScript et TypeScript."
))

add_bullet(doc, (
    "Interface React opérationnelle : liste des projets, détail des logs avec métadonnées complètes "
    "(modèle, tokens, latence, température, PII détectée), tableau de bord Recharts des métriques "
    "agrégées dans le temps."
))

add_bullet(doc, (
    "Modèle de données structuré (MongoDB/Beanie) supportant trois types de projets : LLMObserve "
    "(proxy + métriques), LLMExplain (interprétabilité via records BERT-COLA), LLMInfer, avec "
    "validation stricte de la compatibilité tâche/métrique à la création du projet."
))

# ══════════════════════════════════════════════════════════════════════════════
# 5. Contribution scientifique
# ══════════════════════════════════════════════════════════════════════════════

add_heading1(doc, '5. Contribution scientifique')

add_bullet(doc, (
    "Pipeline d'évaluation LLM asynchrone non bloquant : architecture découplant le chemin "
    "critique de réponse (proxy → fournisseur → client) de l'évaluation computationnellement "
    "coûteuse via asyncio.create_task, garantissant une latence de réponse indépendante de la "
    "complexité des métriques calculées."
),
bold_prefix="Évaluation asynchrone : ")

add_bullet(doc, (
    "Système de sélection dynamique des métriques par LLM (decide_on_metrics) : approche originale "
    "consistant à déléguer au modèle de langage lui-même la décision de quelles métriques sont "
    "pertinentes pour un échange donné, en tenant compte du type de tâche, de la présence de contextes "
    "RAG et du contenu de la réponse."
),
bold_prefix="Sélection dynamique de métriques : ")

add_bullet(doc, (
    "Vérification factuelle par décomposition en affirmations atomiques (claim-based faithfulness) : "
    "méthode de décomposition systématique d'une réponse LLM en affirmations factuelles indépendantes, "
    "vérifiées individuellement contre la connaissance du modèle, produisant un score de fidélité "
    "granulaire et une liste d'hallucinations localisées."
),
bold_prefix="Détection des hallucinations par claims : ")

add_bullet(doc, (
    "Intégration d'un pipeline de protection PII réversible dans un proxy LLM transparent : "
    "anonymisation par NER BERT-large avec Vault de secrets session-scoped, garantissant que "
    "les données personnelles ne transitent jamais en clair vers le fournisseur tout en préservant "
    "la cohérence sémantique de l'échange pour le client."
),
bold_prefix="Anonymisation PII réversible : ")

add_bullet(doc, (
    "Catalogue hétérogène de métriques couvrant des tâches fondamentalement différentes (génération "
    "conversationnelle, RAG, résumé, traduction neuronale, génération de code compilable) au sein "
    "d'un unique moteur d'évaluation paramétrable par type de projet et de tâche."
),
bold_prefix="Catalogue multi-tâches : ")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Perspectives
# ══════════════════════════════════════════════════════════════════════════════

add_heading1(doc, '6. Perspectives')

add_heading2(doc, '6.1 Améliorations futures')

add_bullet(doc, (
    "Fine-tuning du modèle de sélection automatique des métriques (decide_on_metrics) sur un corpus "
    "annoté d'interactions LLM pour améliorer la précision de la sélection sans requérir un appel "
    "LLM supplémentaire à chaque interaction."
))

add_bullet(doc, (
    "Extension du pipeline RAG avec réactivation du module FactChecker (FAISS + BAAI/bge-large-en-v1.5) "
    "pour la vérification factuelle des claims contre une base de connaissances externe, "
    "permettant une validation hybride (connaissance paramétrique + RAG)."
))

add_bullet(doc, (
    "Amélioration de la détection des PII pour couvrir les données structurées non textuelles "
    "(IBAN, numéros de carte, codes postaux, numéros de téléphone internationaux) via des "
    "recognizers regex combinés au NER BERT."
))

add_bullet(doc, (
    "Optimisation de la latence du modèle COMET (Unbabel/wmt22-comet-da) en production CPU "
    "par quantification du modèle ou migration vers une inférence GPU dédiée."
))

add_heading2(doc, '6.2 Extensions possibles')

add_bullet(doc, (
    "Support multi-fournisseurs : extension du proxy pour supporter nativement Anthropic Claude, "
    "Google Gemini et les modèles open-source (Ollama, vLLM) sans modification du schéma de requête."
))

add_bullet(doc, (
    "Alerting en temps réel : mise en place de seuils d'alerte configurables par métrique et par "
    "projet (ex. : alerte si hallucination_rate > 20 %, si moderation_score > 0.7), avec "
    "notification webhook ou e-mail."
))

add_bullet(doc, (
    "Apprentissage continu : capitalisation sur les traces d'interactions évaluées pour construire "
    "des datasets d'entraînement permettant le fine-tuning de modèles d'évaluation spécialisés "
    "par domaine métier."
))

add_bullet(doc, (
    "SDK client natif : développement d'une bibliothèque Python et JavaScript pour simplifier "
    "l'intégration de LLMInsight dans les applications existantes sans reconfiguration du endpoint."
))

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

output_path = 'rapport_cir_llminsight_2025.docx'
doc.save(output_path)
print(f"Rapport généré : {output_path}")
