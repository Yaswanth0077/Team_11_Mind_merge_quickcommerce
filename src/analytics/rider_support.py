"""Non-punitive, testable group-level support hypotheses for PS7 segments."""

SEGMENT_SUPPORT_ACTIONS = {
    0: {
        "support": (
            "Review anonymized dispatch sequences and multi-order batching for this "
            "group; test clearer dispatch/load visibility and route-sequencing support."
        ),
        "training": (
            "Offer voluntary, scenario-based practice on multi-order handoff and route "
            "sequencing; first confirm the workflow is relevant through observation."
        ),
        "roster": (
            "Compare peak-period assignments and concurrent delivery load before "
            "changing rosters; do not reduce opportunities based on segment membership."
        ),
        "kpi": (
            "Group late rate and median/p90 delivery time, stratified by traffic, "
            "distance and multiple-delivery load; report sample sizes."
        ),
        "validation": (
            "Use a time-bounded, opt-in team process pilot with a comparable group; "
            "check uncertainty and workload before and after."
        ),
    },
    1: {
        "support": (
            "Review route and dispatch context for longer-distance orders, including "
            "whether estimated travel distance and assigned service area are reliable."
        ),
        "training": (
            "Offer route-planning and handoff refreshers only where an observed "
            "process review identifies a specific skill or information gap."
        ),
        "roster": (
            "Check whether peak-period coverage and assignments are balanced across "
            "longer routes; compare like-for-like locations and time windows."
        ),
        "kpi": (
            "Distance- and period-stratified late rate, delivery-time median/p90, "
            "and workload distribution."
        ),
        "validation": (
            "Pilot route-information or dispatch changes by comparable route/time "
            "cohorts and retain a control; do not infer rider causes from the profile."
        ),
    },
    2: {
        "support": (
            "Document the group's observed delivery processes and invite voluntary "
            "peer sharing; treat lower aggregate late rate as a prompt for inquiry, "
            "not proof of a best practice."
        ),
        "training": (
            "Use team-led process exchange and the same optional refresher resources "
            "available to all riders; avoid labeling individuals."
        ),
        "roster": (
            "Review whether lower group-average delivery volume reflects scheduling, "
            "assignment mix or observation coverage; balance peak demand without "
            "assuming unused capacity."
        ),
        "kpi": (
            "Late rate and delivery-time median/p90 alongside deliveries per rider, "
            "peak-period workload and sample sufficiency."
        ),
        "validation": (
            "Test any shared workflow on a voluntary, comparable cohort and verify "
            "that the result persists across route and traffic mixes."
        ),
    },
}
