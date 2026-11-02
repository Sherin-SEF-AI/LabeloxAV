# Running it as a team

This page is for whoever sets LabeloxAV up for a small team: adding people, deciding who sees which data,
choosing what each project labels, and letting the assistant take some of the routine work.

## People and roles

An administrator adds a teammate in **Settings > Team**. The app produces a sign-in link to send them; it
works once, expires, and leaves them with an ordinary password. The same button issues a fresh link when
someone forgets theirs.

| Role | Can |
| --- | --- |
| Annotator | label, review their own work, use the assistant, see the projects open to them |
| Reviewer | also accept and reject others' labels, apply bulk edits, approve the assistant's proposals, start training |
| Administrator | everything, including the team, API keys, projects and who sees them |

Scripts and notebooks use an API key from **Settings > API keys**, not a person's password.

## Projects: who sees what

Every session belongs to one project. A project is visible to the **whole team** (the default) or to its
**members only**. For a members-only project, everyone who is not a member or an administrator stops seeing
its sessions, frames and labels everywhere: in lists, in search, and by direct link, where the answer is
"not found" so nothing confirms the data exists.

Set it up in **Settings > Projects**: create the project, make it members only, add its members, and move its
sessions in. An upload or import can name its project; without one it goes to the default project.

Two things are not scoped by project: aggregate charts (they count across all projects but name no item),
and the corpus-wide engines in Labs, which are for administrators.

## What each project labels

The ontology is shared, but a project chooses which classes it labels, under **Settings > Projects > Classes**:

- **Indian roads**: every class, including autorickshaws, cycle rickshaws, handcarts and cattle. The default.
- **Generic driving**: the road users and infrastructure found anywhere, without the Indian long tail.
- **Custom**: class by class.

The editor then offers only those classes (1-9 hotkeys included), auto-labelling writes no others, and a
training run scoped to the project learns those classes. Labels already saved keep their class. Ignore
regions and your team's own custom classes are in every set.

## The assistant

**Assistant** in the sidebar takes questions in words: "how many cattle are waiting for review in the Delhi
drive", "which drives do we have from Pune", "look for label mistakes in this session". It answers from the
data and shows which lookups it made.

It cannot change anything itself. When you ask for a change (reclassify, send labels back to review,
auto-label a session, run the error detectors), it creates a **proposal** with exactly what it would touch.
A reviewer approves or rejects it on the same page. Approving runs the change the way a person would: class
changes go back to review and can be reverted, human labels are never touched, and auto-labelling waits its
turn for the GPU like any other job.

By default the model runs on your own machine (Ollama, `qwen2.5:7b`), and only text is exchanged: counts,
class names and ids, never an image. To use a hosted model instead, set `LBX_ASSISTANT__PROVIDER` to `groq`
or `anthropic` with that provider's key. The assistant page shows which model answers.

## Knowing whether the clever parts help

Each assistive feature can be measured from the product, against gold or a planted ground truth:

- **My queue > Why this order?** shows how much faster the queue's order finds wrong labels than random.
- **Label errors > Do the detectors find real errors?** (administrators) plants wrong classes, measures
  which detectors catch them, and undoes everything.
- The assistant's tool choice is scored by `POST /api/assistant/eval`.

Current numbers and what they mean are in [Measurement](../MEASUREMENT.md#is-each-assistive-feature-earning-its-place).
