# `enable_forms`

```python
enable_forms: bool = False
```

**Category**: Artifacts

## What it does

Enables `<lollms_form>` — the model can emit an interactive form descriptor
(fields, ratings, textareas, sections) that the host UI renders natively.

## What to expect

- Form instructions are injected **only when trigger keywords** appear in the
  user message ("form", "formulaire", "survey", "questionnaire").
- The parsed descriptor is parked in `discussion._get_pending_forms()[form_id]`
  and emitted as `MSG_TYPE_FORM_READY`.

## Recovering generated stuff (answers)

Call `discussion.submit_form_response(form_id, answers)` — answers are formatted
and injected into the conversation as a user message, and a
`MSG_TYPE_FORM_SUBMITTED` event fires.

## Usage

```python
result = discussion.chat("Ask me a survey with 3 questions", enable_forms=True)
forms = discussion._get_pending_forms()
# ... render in your UI, collect answers ...
discussion.submit_form_response(form_id, answers)
```

## Related flags

- `enable_artefacts` — master switch.