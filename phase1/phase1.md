---
marp: true
paginate: true
style: |
  section {
    padding: 24px;
    font-size: 12px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
  }
  section h1 {
    font-size: 24px;
  }
  section h2 {
    font-size: 18px;
  }
  section table {
    font-size: 12px;
    margin-inline: auto;
  }
  section > ul {
    width: max-content;
    max-width: 92%;
    text-align: left;
  }
  section.human-layout {
    display: block;
    text-align: center;
  }
  section.human-layout .human-columns {
    display: grid;
    grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
    align-items: center;
    gap: 24px;
    text-align: left;
  }
  section.human-layout .human-copy table {
    width: 100%;
  }
  section.human-layout .human-visual img {
    display: block;
    width: 100%;
    height: auto;
  }
  section.human-layout .human-visual img + img {
    margin-top: 12px;
  }
---

# Problem

To detect ambiguity in Natural-Language Software Requirements using the RoBERTa model

# Current Progress 

- created a common fine-tuned pipeline for RoBERTa using the annotated PURE dataset, saved the fine-tuned RoBERTa to be used by the group
- analyzed the training dataset (Human / AI-Aided)
- analyzed the test dataset and corresponding results from the fine-tuned RoBERTa

---

<!-- _class: human-layout -->
# Human Group

<div class="human-columns">
<div class="human-copy">
<table>
<tbody><tr><td>K12148218: Ostap Ohlashennyy</td><td>K12434880: Lloyd Aaron Yandoc</td></tr></tbody>
</table>
<h2>Key Findings</h2>
<ul>
<li>The label mix is generally balanced across the training, validation, and test sets.</li>
<li><code>weak_modal</code> is the most frequent annotated ambiguity reason.</li>
<li>Model probabilities are often near 0 or 1, including for some incorrect predictions.</li>
<li><code>multiple_actions</code> has the highest observed miss rate.</li>
</ul>
</div>
<div class="human-visual">
<img src="phase1_human_viz.png" alt="Human Group visualizations">
</div>
</div>

---

<!-- _class: human-layout -->
# AI-Aided Group

<div class="human-columns">
<div class="human-copy">
<table>
<tbody><tr><td>K12147981: Vladyslav Protsko</td><td>K12111267: Simon Stromberger</td></tr></tbody>
</table>
<h2>Key Findings</h2>
<ul>
<li>The model missed 25/28 requirements annotated with multiple actions alone, versus 0/26 with additional reasons. The contrast persists within shared-source comparisons and across length bands.</li>
<li>Clarifications noticeably lowered ambiguity scores for two of four requirements, while paraphrases barely affected them. Only the broader size/timing rewrite changed the predicted class. This small pilot shows wording sensitivity, not reliable improvement.</li>
</ul>
</div>
<div class="human-visual">
<img src="phase1_ai_fig_1a.png" alt="Miss rates: multiple actions only vs. with additional reasons">
<img src="phase1_ai_fig_6.png" alt="Paired score changes: clarification versus control">
</div>
</div>

---