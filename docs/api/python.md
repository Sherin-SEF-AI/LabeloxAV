# Python reference

The seams worth programming against, not every module. LabeloxAV has roughly 600 Python modules; auto-dumping
all of them produces a reference nobody can navigate and that goes stale the moment an internal helper moves.
What follows is the surface that is deliberately stable: the domain-pack contract, the engine's bridge to it,
and the measurement primitives.

## The domain pack contract

A pack is how one engine serves two domains. `packs/av` is autonomous driving; `packs/sec` is physical
security. The engine never imports either: `.importlinter` forbids `core`, `services` and `db` from
importing `packs.av` or `packs.sec`, and everything goes through the registry.

::: packs.base.DomainPack
    options: {show_root_heading: true, members: false}

::: packs.base.OntologySpec

::: packs.base.CliqueSpec

::: packs.base.RelationSpec

::: packs.base.TrackEventSpec

::: packs.base.ContextSpec

::: packs.base.RegionSpec

## The engine's bridge

Every per-domain question the engine asks goes through here, so a second domain behaves differently at
runtime rather than only existing in the registry.

::: services.domain
    options: {members: [active_pack, pack_for_session, safety_l1, critical_class_names, critical_class_ids, context_spec, validate_context, track_event_spec, validate_track_event_type, class_aliases]}

## Ontology

::: services.autolabel.ontology.Ontology
    options: {members: [by_id, by_name, has_name, attrs_for_class, aliases_for, validate_attrs, derive_attrs]}

## The confidence gate

Where humans enter. Read the module docstring before quoting any threshold: a threshold is a precision floor
only for a class with a fitted operating point, and the configured constants are not that.

::: services.autolabel.gate
    options: {members: [gate_object, needs_vlm, vlm_confirmed]}

## Measurement

The judge, its calibration, and the correction that makes a machine-derived precision quotable.

::: services.labelops.vlm_review
    options: {members: [build_judge_prompt, parse_judge_reply, judge_objects, prereview_batch, judge_agreement, judged_precision]}

::: services.labelops.class_precision
    options: {members: [sample_class, judge_class, class_targets]}

::: services.labelops.sampling
    options: {members: [wilson_interval, rogan_gladen, rogan_gladen_interval]}

## Temporal

Filling frames between detections, and refusing to when the anchors are not one object.

::: services.temporal.gap_gate
    options: {members: [same_object, is_discontinuity, GateResult]}

::: services.temporal.interpolate
    options: {members: [interpolate_track_keyframed, build_box_interpolator]}

## Reversibility

Every corpus-wide write in this system is undoable through one of these.

::: services.review_apply
    options: {members: [apply_review_batch]}

::: services.review_batch
    options: {members: [change_record, record_batch, revert_batch]}

::: services.agent.runs
    options: {members: [revert_run]}
