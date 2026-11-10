# Phase I: multiple-actions investigation — AI interaction record

**Tool:** Codex, GPT-6 Astra  
**Scope:** brainstorming, implementation, interpretation, and refinement of the [multiple-actions notebook](../phase1/AI-aided/multiple_actions_investigation.ipynb).

## 1. Choosing a focused investigation

**Request:** I asked the assistant to inspect the course instructions and my AI-aided teammate's existing notebooks to identify useful further work and ideas.

**AI contribution:** proposed ambiguity-category comparisons, keyword-versus-context behavior, source differences, confident errors, paraphrase stability, and clarification experiments. After reviewing the teammate's overview, the assistant recommended investigating an existing finding more deeply instead of adding another general performance report.

The teammate's notebook identified 25 misses among 54 requirements annotated with `multiple_actions`. The assistant checked the saved predictions and subdivided these examples:

| Annotation group | Missed / total |
|---|---:|
| `multiple_actions` only | 25/28 |
| `multiple_actions` plus other reasons | 0/26 |

**Decision:** I chose this contrast as the focus, with source and length comparisons as supporting checks. I later refined the question to **How does RoBERTa’s detection differ when `multiple_actions` is annotated alone versus alongside other reasons?** This wording describes the annotation comparison without implying a causal effect. I deferred a separate investigation of false positives in long requirements and did not develop the other brainstormed questions as separate analyses in this notebook.

The contrast suggested possible reliance on additional wording cues, but was treated as a hypothesis, not evidence of the model's internal mechanism.

## 2. Main notebook-generation prompt

I used the following prompt to request the initial notebook implementation. It is preserved as originally written; later revisions are described below:

> Create `multiple_actions_investigation.ipynb` in our team repository. Read the existing notebooks and prediction code first, and preserve them.
>
> Question: **How does the presence of additional ambiguity cues affect RoBERTa’s detection of requirements annotated with multiple actions?**
>
> Implement:
>
> - Verify the contrast: 25/28 missed when `multiple_actions` is the only reason versus 0/26 with additional reasons.
> - Compare error rates, sample sizes, uncertainty, and ambiguity-score distributions. Investigate differences in source, length, and additional reasons.
> - Build informative interactive visualizations appropriate for our AI-aided pair: linked filters, example selection with complete text, and subgroup comparisons. Choose interaction that helps investigate the findings, while keeping the notebook manageable.
> - Prepare draft clarification edits that remove additional ambiguity cues while preserving the multiple-action structure, plus harmless paraphrases as controls. Record their purpose and leave them pending human review.
> - Provide optional inference on approved edits using our frozen model, with interactive original-versus-edited score comparisons and prediction-flip summaries. Keep inference disabled by default and preserve existing results.
> - Explain findings and limitations.
>
> Keep this a focused student project. Phase I uses only inputs and outputs; multiple actions do not automatically imply ambiguity. Document any added dependencies. Verify using saved predictions, without running training or new inference.

**AI contribution:** implemented subgroup plots, approximate Wilson intervals, source/length checks, an example explorer, draft edits, and optional frozen-model inference. Further checks found 22/25 versus 0/20 misses within the six sources containing both groups; both overall groups had a median length of 26.5 words. These checks strengthened the reason to investigate the contrast, without isolating its cause.

## 3. Revising visualization choices

**Representative prompt:** “You implemented only interactive plots. Do all of them provide useful additional insights that normal, non-interactive plots could not show? I don't think so. Which plots would you recommend to make static?”

The assistant initially made many plots interactive. I questioned whether hover and zoom added enough value to justify the complexity.

**Decision and revision:** sections 1–2 became static, with clearer labels, sample counts, and concise interpretations. In section 6, the score-change box plot became a static paired comparison connecting each original's clarification and control. The original-versus-edited scatter first became static; I later agreed to remove it entirely because it duplicated the paired comparison and the example selector. I also accepted replacing the additional-reason plots in section 2c with a compact count table, keeping the notebook focused alongside my teammate’s overview.

I retained interactivity for tasks where selection changes what can be investigated: filtering requirements, reading complete text, reviewing drafts, and comparing an original with its edits. This was a refinement of the AI's original design, not acceptance of every generated visualization.

## 4. Adding interaction that supports interpretation

I discussed two further ways to improve interpretation: linking source comparisons to individual requirements and highlighting changed wording alongside predictions.

**Decision:** I kept section 2 static and accepted an **Explore Pontis** shortcut in section 3 instead of cross-section clicking. The shortcut resets other filters to show both groups, and point clicks select the full requirement text. I retained the dropdown as an alternative.

For section 6, the assistant implemented side-by-side word differences with removed/added words highlighted, together with scores, predictions, expected labels, and review notes. This makes the difference between a broad rewrite and a one-word modal edit visible. The saved pilot has one prediction flip, following the size/timing rewrite; highlighting helps prevent attributing that result to clarification in general. The highlights represent text differences, not model attribution.

## 5. Verification and revision effort

The generated interactive views required corrections for rendering problems and duplicated content after repeated execution. The assistant initially replaced interaction with static fallbacks, but I requested restoring the useful interactive features. This showed that successful code execution alone was insufficient: the views also needed checking in the actual notebook environment and across repeated runs.



AI assistance accelerated implementation, especially of very extensive and complicated interative plots, but also introduced additional verification and editing work.
