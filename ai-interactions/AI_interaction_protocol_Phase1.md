# AI Interaction Protocol (dataset-context-model-behaviour.ipynb)

## Overview
AI assistance was used throughout the Phase I project as a support for exploring analytical options, developing visualizations, improving notebook organization, and refining written explanations. The interaction was iterative: suggestions were requested for specific parts of the work, reviewed against the project requirements and available results, and then accepted, adapted, or left out.

## Main areas of AI assistance

### 1. Proposed directions for analysis
- Suggested moving beyond dataset description to examine the behaviour of the ambiguity classifier using its predictions and errors.
- Proposed investigating differences between false positives and false negatives, confidence in incorrect predictions, requirement length, source-document variation, and annotated ambiguity categories.
- Highlighted multiple-action ambiguity as a potentially useful focus for a deeper investigation because of its observed miss rate.
- Suggested treating these patterns as exploratory findings and checking sample sizes and alternative explanations before drawing conclusions.

### 2. Proposed visualizations
- Suggested a confusion matrix to summarize the types of classification errors.
- Proposed confidence and calibration plots to examine whether high predicted confidence corresponds to correct predictions.
- Suggested comparing false-positive and false-negative rates across requirement-length groups.
- Proposed subgroup comparisons by ambiguity reason and source document.
- Suggested probability-distribution plots to show how well the model separates the two annotated classes.
- Also proposed threshold-sensitivity and reason co-occurrence plots as additional exploratory views.

The visualizations were not all treated as equally important. Redundant or less relevant plots were later deprioritized so that the final analysis would remain focused on the main research questions.

### 3. Proposed notebook structure
- Suggested separating the broad overview of dataset context and model behaviour from the more focused investigation of multiple-action ambiguity.
- Recommended keeping a compact set of core figures in the overview, supported by a small number of representative examples.
- Suggested moving optional inference and wording-edit experiments into clearly marked sections, rather than making them prerequisites for reading the analysis.
- Recommended removing repeated plots and analyses that did not contribute directly to the central narrative.

### 4. Descriptions and interpretations
- Drafted and refined explanatory text for visualizations, including the analytical question, what the plot displays, the main observed pattern, and relevant limitations.
- Helped phrase interpretations of model errors, confidence, length-related patterns, source variation, and ambiguity-reason subgroups.
- Added cautions around small subgroup sizes, overlapping annotations, and the difference between association and causation.

### 5. Implementation and review support
- Assisted with drafting and extending notebook code for additional visualizations.
- Helped inspect the training metadata, saved test predictions, metrics, and training history to establish which outputs were available for analysis.
- Reviewed the proposed notebook contents for duplication and alignment with the Phase I objective.
- Provided checks on code structure and syntax; execution in the target project environment remained necessary to verify the final notebook outputs.

## Suggestions adopted, adapted, or deprioritized

**Adopted**
- A model-behaviour focus alongside dataset context.
- Analysis of false positives, false negatives, confidence, requirement length, source variation, and ambiguity-reason subgroups.
- Plot descriptions and interpretation text that explain both findings and limitations.
- A two-part notebook narrative: an overview followed by a focused multiple-actions investigation.

**Adapted**
- The initial set of proposed visualizations was narrowed to avoid repetition and keep the analysis question-driven.
- Confidence analysis was extended toward calibration, while retaining the distinction between predicted confidence and actual correctness.
- Training details were reduced to supporting context rather than a central result.
- The multiple-actions finding was used to motivate further investigation, while being described cautiously because of the subgroup size.

**Deprioritized**
- Repeated confidence visualizations and general plots that duplicated information already shown elsewhere.
- Threshold-sensitivity and general reason co-occurrence analyses where they did not strengthen the main narrative.
- Long lists of examples and descriptive charts without a clear follow-up analytical question.

## Limitations and failure patterns in AI-assisted work
- **Generated code required verification:** syntax checks did not guarantee successful execution with the project's files, dependencies, and environment.
- **Risk of unnecessary complexity:** broad requests for additional plots produced more possible visualizations than were needed. Relevance and redundancy had to be reviewed.
- **Risk of overinterpretation:** subgroup differences could be affected by small sample sizes, multi-label annotations, and source composition. They should not be presented as causal findings.
- **Risk of imprecise terminology:** confidence, calibration, precision, recall, and error rates needed to be described distinctly.

## Emerging best practices
1. Use AI for specific tasks such as proposing analytical directions, suggesting plot types, drafting code, or refining explanations.
2. Provide the project brief, data schema, notebook, and actual model outputs so suggestions are grounded in the work.
3. Make each visualization answer a defined question and include sample sizes, denominators, and interpretation caveats.
4. Treat AI-proposed patterns as hypotheses to verify against the data and representative examples.
5. Review proposed plots for redundancy; more visualizations do not necessarily produce a stronger analysis.
6. Keep the analytical narrative focused on findings that are supported by the available evidence.

## Overall assessment
AI assistance contributed ideas for analytical direction, visualization design, notebook structure, implementation, and written interpretation. Its suggestions were most useful when applied to specific questions and reviewed for relevance. The resulting analysis required selection and refinement to ensure that the plots, descriptions, and conclusions remained consistent with the project brief and the available evidence.
