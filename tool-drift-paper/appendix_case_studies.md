# Appendix Case Studies

Public repository:
[AniruddhaChattopadhyay/tool-drift](https://github.com/AniruddhaChattopadhyay/tool-drift)

All repo-relative paths below are local monorepo paths (`tool-drift/...`).
Each source run also includes the corresponding public GitHub blob link so
readers can inspect the exact JSON and code used in these appendix examples.

## DICE Example: `dice_round_1_36`

Source run:
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7423`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7423))

### Why this example is useful

This is a clean illustration of the paper's main non-oracle failure mode:

- the model is correct on the original interface,
- the drifted call loses the tool name entirely,
- the candidate-set retry repair recovers the correct tool and drifted arguments.

### Task

The dialogue asks the model to schedule a haircut appointment with:

- `service = haircut`
- `date = 11-10`
- `time = 10:00`
- `location = 456 Salon Ave.`

See the prompt and task record at
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7424`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7424)).

### Original interface and original prediction

The original tool is `schedule_appointment` with fields:

- `service`
- `date`
- `time`
- `location`

See the original schema at
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7426`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7426)).

The model's original call is correct:

```json
{
  "name": "schedule_appointment",
  "arguments": {
    "service": "haircut",
    "date": "11-10",
    "time": "10:00",
    "location": "456 Salon Ave."
  }
}
```

See:
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7499`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7499))
and the recorded match outcome at
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7634`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7634)).

### Drifted interface

The drifted tool keeps the same tool name but renames fields:

- `date -> date_v2`
- `time -> time_v2`
- `location -> location_v2`

It also adds a legacy note in the description mentioning older payloads, and the
visible candidate set includes a distractor tool:

- `schedule_appointment`
- `schedule_appointment_alt_1`

See:

- drifted schema:
  `tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7457`
  ([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7457))
- visible candidate tools:
  `tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7495`
  ([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7495))

### Drifted prediction

The drifted prediction is:

```json
{
  "name": "",
  "arguments": {}
}
```

See:
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7621`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7621))

The validator records this as:

- `missing_tool_name`

See:
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7508`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7508))

### What the empty `pred_call` means

In this codebase, an empty call does **not** mean the benchmark itself supplied an
empty gold answer. It means the inference client failed to recover any usable tool
call from the model response.

The extraction logic is in
`tool-drift/inference/openrouter_client.py:112`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/inference/openrouter_client.py#L112)).
It returns:

```python
{"name": "", "arguments": {}}
```

when:

- the provider returns no `choices`, or
- the provider returns a message with no `tool_calls`, and
- the message content does not contain parseable JSON with `name` and `arguments`.

See:

- no choices fallback:
  `tool-drift/inference/openrouter_client.py:118`
  ([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/inference/openrouter_client.py#L118))
- missing `tool_calls` / JSON fallback:
  `tool-drift/inference/openrouter_client.py:123`
  ([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/inference/openrouter_client.py#L123))
- final empty return:
  `tool-drift/inference/openrouter_client.py:145`
  ([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/inference/openrouter_client.py#L145))

Important limitation: this run artifact does **not** store the raw drifted model
response payload for the original failed prediction, so we cannot tell from the JSON
whether the model:

- produced plain text instead of a tool call,
- produced malformed JSON,
- produced a tool call in an unsupported shape, or
- returned no usable tool output at all.

The strongest defensible statement is:

> The drifted prediction yielded no extractable tool call, so the runtime recorded
> `pred_call = {"name": "", "arguments": {}}`.

This interpretation is also supported by the fact that the naive retry and tool-list
reprompt baselines for this same example also remained empty:

- naive retry:
  `tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7675`
  ([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7675))

### Repair behavior

This example uses:

- `repair_strategy = "candidate_tool_retry"`

See:
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7523`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7523))

The repair prompt explicitly lists the two visible tools and asks the model to choose
exactly one and infer the missing required fields from the task.

See the stored repair prompt at
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7534`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7534)).

The repaired call is:

```json
{
  "name": "schedule_appointment",
  "arguments": {
    "service": "haircut",
    "date_v2": "11-10",
    "time_v2": "10:00",
    "location_v2": "456 Salon Ave."
  }
}
```

See:
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7625`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7625))

The repaired call matches the drifted gold call:

- `repaired_match.matched = true`

See:
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:7653`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L7653))

### Appendix-ready takeaway

Suggested one-paragraph summary:

> In DICE example `dice_round_1_36`, the base model correctly called
> `schedule_appointment` on the original interface, but under drift it produced no
> extractable tool call at all (`pred_call = {"name": "", "arguments": {}}`).
> The drifted schema had renamed `date`, `time`, and `location` to `*_v2` fields and
> introduced a distractor candidate tool. The validator flagged this as
> `missing_tool_name`, after which the non-oracle candidate-set retry recovered the
> correct tool and arguments, yielding a fully matched repaired call. This example is
> representative of the paper's main non-oracle bottleneck: missing tool names under
> candidate-set drift.

## DICE Example: `dice_round_1_9` — forced-tool repair fills in renamed fields

Source run:
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:1816`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L1816))

### Why this example is useful

This is a clean example of the **non-empty, partially-compliant** drift failure
mode, which is complementary to `dice_round_1_36` (fully empty call). Here the
model does name the correct tool and does adopt *one* of the renamed fields, but
silently drops the rest. The validator catches the missing required `_v2`
fields, and `forced_tool_call` repair recovers them.

### Task

Multi-turn dialogue asking for the top-10 MLB pitchers ranked by ERA for the
2023 regular season.

### Drifted interface

- Tool: `pitching_leaders` (unchanged).
- Schema drift: `sort_column`, `season`, `sports_code`, `game_type` renamed to
  `*_v2`. The `results` field keeps its original name (partial rename).
- Candidate-set drift: distractor `pitching_leaders_alt_1` added.

### Drifted prediction

```json
{"name": "pitching_leaders", "arguments": {}}
```

The model picks the right tool but emits an empty argument object.

Validation flags four `missing_field` errors on the `_v2` fields.

### Repair

- `repair_strategy = "forced_tool_call"` — the provider is forced to call the
  already-resolved tool, and the repair prompt carries the canonical card plus
  the four missing-field errors.

Repaired call:

```json
{
  "name": "pitching_leaders",
  "arguments": {
    "results": 10,
    "sort_column_v2": "ERA",
    "season_v2": "2023",
    "sports_code_v2": "MLB",
    "game_type_v2": "regular season"
  }
}
```

This matches the drifted gold call — `repaired_match.matched = true`.

### Takeaway

When the tool name survives drift but the arguments collapse, structured
validation errors plus forced-tool repair are sufficient: the candidate-set
retry stage is skipped entirely, and a single constrained repair call recovers
every renamed field.

## DICE Example: `dice_round_1_41` — minimal single-field miss

Source run:
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:8411`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L8411))

### Why this example is useful

Demonstrates that repair is triggered even when the model is only one required
field away from a valid call. The value is not in diagnosing a catastrophic
failure but in showing that the validator fires on *any* structural gap, not
just empty calls.

### Task

Retrieve likes for Instagram post `Ck3Yz1nLgX5`, returning 25 results starting
from cursor `QW5vdGhlclBhZ2U=`.

### Drifted interface

- Tool: `get_instagram_post_likes` (unchanged).
- Schema drift: `count -> count_v2`, `end_cursor -> end_cursor_v2`; `shortcode`
  retained.
- Candidate-set drift: `get_instagram_post_likes_alt_1` distractor added.

### Drifted prediction

```json
{
  "name": "get_instagram_post_likes",
  "arguments": {"shortcode": "Ck3Yz1nLgX5", "count_v2": 25}
}
```

The model correctly adopts the renamed `count_v2`, but omits the required
`end_cursor_v2`. Validation: one `missing_field` error.

### Repair

- `repair_strategy = "forced_tool_call"`.

Repaired call adds the missing field from the dialogue context:

```json
{
  "name": "get_instagram_post_likes",
  "arguments": {
    "shortcode": "Ck3Yz1nLgX5",
    "count_v2": 25,
    "end_cursor_v2": "QW5vdGhlclBhZ2U="
  }
}
```

`repaired_match.matched = true`.

### Takeaway

The validator's granularity matters: a single missing required field is enough
to gate repair, and the repair prompt's explicit instruction to "infer all
required fields from the dialogue" is enough to fill the gap.

## DICE Example: `dice_round_1_215` — drift that escapes validation

Source run:
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:38623`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L38623))

### Why this example is useful

This is an **honest limitation case**. The drifted prediction is structurally
valid (correct tool, all required `_v2` fields present, types match), so the
validator marks it `valid = true` and **repair never fires**
(`repair_strategy = "not_needed"`). But the argument value is semantically
wrong, so the drifted call still mismatches the drifted gold.

This example scopes the paper's claim: SchemaShield-Lite recovers
*structure-level* drift failures, not value-fidelity failures.

### Task

Play `Bohemian Rhapsody` on a `JBL Speaker`.

### Drifted interface

- Tool: `play_music` (unchanged).
- Schema drift: `device_name -> device_name_v2`.
- Description drift: "Legacy integration note: older examples may still show
  payloads like `{\"device_name\": \"value\"}`."
- Candidate-set drift: `play_music_alt_1` distractor added.

### Drifted prediction

```json
{
  "name": "play_music",
  "arguments": {"music_name": "Bohemian Rhapsody", "device_name_v2": "JBL"}
}
```

Drifted gold:

```json
{
  "name": "play_music",
  "arguments": {
    "music_name": "Bohemian Rhapsody",
    "device_name_v2": "JBL Speaker"
  }
}
```

Validation: `valid = true`. No repair. `drifted_match.matched = false` because
`"JBL" != "JBL Speaker"`.

### Takeaway

Structural validation is a necessary gate for a safe repair layer — you do not
want repair triggering on calls that look fine — but it is not sufficient to
catch every drift-induced miss. Value-level semantic mistakes (abbreviations,
normalization, unit confusion) pass through unchanged. This is consistent with
the paper's framing of SchemaShield-Lite as a robustness layer for surface-form
validity rather than a general-purpose correctness oracle.

## DICE Example: `dice_round_1_139` — repair fires but partially fails

Source run:
`tool-drift/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json:24886`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/dice/dice-bench-live-20260414-170240-443375/dice_results.json#L24886))

### Why this example is useful

Same tool family as `dice_round_1_9` (`pitching_leaders`) but ends in
`repaired_match = false`. Shows that forced-tool repair can fix missing
*structure* yet still emit *values* that diverge from the gold when the
dialogue is ambiguous or the model normalizes inconsistently.

### Task

Casual multi-turn about MLB pitcher ERA; the task signals `era` and
`regular` season but the drifted gold expects `"ERA"` (uppercase) and
`"regular season"` (full phrase).

### Drifted prediction and repair

Drifted prediction (invalid, missing three required fields):

```json
{
  "name": "pitching_leaders",
  "arguments": {"results": 10, "sort_column_v2": "era"}
}
```

Repaired call (`forced_tool_call`):

```json
{
  "name": "pitching_leaders",
  "arguments": {
    "results": 10,
    "sort_column_v2": "era",
    "season_v2": "2023",
    "sports_code_v2": "mlb",
    "game_type_v2": "regular"
  }
}
```

Drifted gold:

```json
{
  "name": "pitching_leaders",
  "arguments": {
    "results": "10",
    "sort_column_v2": "ERA",
    "season_v2": "2023",
    "sports_code_v2": "mlb",
    "game_type_v2": "regular season"
  }
}
```

Repair filled every missing `_v2` field, so the call is now structurally valid,
but the casing of `sort_column_v2` and the phrasing of `game_type_v2` still
disagree with the gold.

### Takeaway

Counts as an originally-correct slice failure that repair did not rescue. This
example illustrates that the remaining 2/22 unrepaired DICE drift misses in the
final primary run are not adversarial failures of the repair stack — they are
value-fidelity cases that any structure-only repair layer will leave on the
table.

## BFCL Example: `simple_python_28` — candidate retry under heavier distractor pressure

Source run:
`tool-drift/outputs/bfcl/bfcl-live-20260414-170241-830717/bfcl_results.json:2875`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/bfcl/bfcl-live-20260414-170241-830717/bfcl_results.json#L2875))

### Why this example is useful

Shows the `candidate_tool_retry` pattern from `dice_round_1_36` reproducing on
BFCL under the medium-severity drift configuration (three distractors instead
of one), which is the harder setting.

### Task

Compute displacement given initial velocity 10, acceleration 9.8, and time 5.

### Drifted interface

- Tool: `calculate_displacement` (unchanged).
- Schema drift: parameters renamed to `initial_velocity_v2`, `time_v2_v2`,
  `acceleration_v2_v2` (the `_v2_v2` suffixes come from composed rename steps).
- Candidate-set drift: three distractors —
  `calculate_displacement_alt_1/2/3` — added alongside the true tool.

### Drifted prediction

```json
{"name": "", "arguments": {}}
```

Validator: `missing_tool_name` across the four visible candidates.

### Repair

- `repair_strategy = "candidate_tool_retry"` — first resolves a concrete tool
  from the four visible candidates, then issues a constrained repair call
  against that tool.

Repaired call:

```json
{
  "name": "calculate_displacement",
  "arguments": {
    "initial_velocity_v2": 10,
    "time_v2_v2": 5,
    "acceleration_v2_v2": 9.8
  }
}
```

Matches the drifted gold exactly — `repaired_match.matched = true`.

### Takeaway

The non-oracle candidate retry continues to work under BFCL's heavier
candidate-set drift. The same two-stage pattern — resolve visible tool, then
constrained repair — that drives the DICE recovery also explains the BFCL
gain from 0.580 to 0.805.

## BFCL Example: `simple_python_27` — composed schema drift, one field survives rename

Source run:
`tool-drift/outputs/bfcl/bfcl-live-20260414-170241-830717/bfcl_results.json:2585`
([GitHub](https://github.com/AniruddhaChattopadhyay/tool-drift/blob/main/outputs/bfcl/bfcl-live-20260414-170241-830717/bfcl_results.json#L2585))

### Why this example is useful

A forced-tool repair case on BFCL where the drifted schema shows the effect of
composed rename operators: fields already renamed once get renamed again,
producing `_v2_v2` suffixes. The model latches onto the single still-single-
suffixed field and drops the rest.

### Task

Compute final velocity given initial velocity 10, acceleration 2, and time 5.

### Drifted interface

- Tool: `final_velocity` (unchanged).
- Schema drift (composed): `initial_velocity_v2`, `acceleration_v2_v2`,
  `time_v2_v2`.
- Candidate-set drift: three distractors added.

### Drifted prediction

```json
{
  "name": "final_velocity",
  "arguments": {"initial_velocity_v2": 10}
}
```

Validation flags both `_v2_v2` fields as missing.

### Repair

`repair_strategy = "forced_tool_call"`. Repaired call:

```json
{
  "name": "final_velocity",
  "arguments": {
    "initial_velocity_v2": 10,
    "acceleration_v2_v2": 2,
    "time_v2_v2": 5
  }
}
```

Matches the drifted gold — `repaired_match.matched = true`.

### Takeaway

Even under composed schema edits that produce unusual-looking suffixes, the
canonical tool card in the repair prompt presents the current field names
plainly, and forced-tool repair recovers the full argument set. This is the
BFCL analogue of `dice_round_1_9`.

## Coverage summary

| Example | Benchmark | Strategy | Failure mode | Outcome |
| --- | --- | --- | --- | --- |
| `dice_round_1_36` | DICE | `candidate_tool_retry` | empty pred, missing tool name | repaired match |
| `dice_round_1_9` | DICE | `forced_tool_call` | partial args, 4 missing `_v2` fields | repaired match |
| `dice_round_1_41` | DICE | `forced_tool_call` | one missing required field | repaired match |
| `dice_round_1_215` | DICE | `not_needed` | value-level drift escapes validator | unrepaired (limitation) |
| `dice_round_1_139` | DICE | `forced_tool_call` | repair fills structure, values still wrong | unrepaired (limitation) |
| `simple_python_28` | BFCL | `candidate_tool_retry` | empty pred, 4-way candidate list | repaired match |
| `simple_python_27` | BFCL | `forced_tool_call` | partial args under composed rename | repaired match |

The first group (rows 1–3, 6–7) illustrates the paper's positive claims. The
second group (rows 4–5) bounds them: SchemaShield-Lite is a structural repair
layer, so drift failures that survive structural validation or that encode
value-level ambiguity in the dialogue are not addressed by this repair stack.
