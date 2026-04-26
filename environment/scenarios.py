"""Scenario templates and procedural generation for Project Buren."""

from __future__ import annotations

import random
import re
from typing import Callable

from environment.state import BurenState, phase_from_age

# --- At least 40 messy first-person templates with placeholders ---

SCENARIO_TEMPLATES: list[str] = [
    # Early phase (20s–30s)
    "You're {age} and your manager just pulled you aside. There's a {role} opening at a {company_type} in {city} — more money ({salary}k), but you'd be traveling 3 weeks a month. {family_situation}. You have until Friday to decide.",
    "It's 2am and you can't sleep. Your {age}-year-old body isn't recovering from workouts the way it used to. Your doctor mentioned {health_concern} at your last checkup. Your calendar has zero free days this month.",
    "Roommate drama: {city} rent went up again and {family_situation}. You're {age}, juggling {side_hustle} on nights you should be sleeping. Someone says you should just move farther out — but the commute would wreck your mornings.",
    "Your partner texts: they want to {relationship_move}. You're {age}, still paying {debt_level} from school, and {company_type} just announced {role} layoffs. You love them but the timing feels insane.",
    "A friend dangles a startup idea — {risky_opportunity}. You'd sink {salary}k of savings and probably lose weekends for a year. {family_situation}. You're {age} and terrified of being left behind.",
    "Therapist cancelled again. Work wants you on a {city} client next week. You're {age}, running on coffee, and {health_concern} keeps nagging you. Do you finally say no to the trip?",
    "Parents keep asking when you'll 'get serious' about {career_track}. You're {age}, actually liking your messy {role} job, but {debt_level} looms. Do you take the safer corporate offer?",
    "Gym membership auto-renewed. Bank account says no. You're {age}, {side_hustle} is picking up, and your boss wants overtime for a {company_type} launch. Skip health or skip cash?",
    "Dating app match wants to move fast — move in, split rent in {city}. You're {age}, {family_situation}, and your lease isn't up. Feels romantic and reckless at once.",
    "You bombed a presentation. Boss says take the weekend — but there's a {role} certification exam Monday you didn't study for. {health_concern}. You're {age}; pushing through is your default.",
    "Old college friend got rich on crypto and offers to front you into {risky_opportunity}. {family_situation}. You're {age} and embarrassed you still rent.",
    "Doctor says cut stress or {health_concern} gets worse. Your {company_type} team is understaffed and you're the only one who knows the codebase. You're {age}.",
    "You're {age}, {city} winter, and considering grad school — more {debt_level}, maybe better {career_track}. Partner thinks you should wait. You don't know who is right.",
    "Friday night: friends want a trip; you have a {side_hustle} deadline. {family_situation}. You're {age} and feel like you're always the one who says no.",
    "HR emailed about a wellness program. Feels performative. You're {age}, averaging 5 hours sleep, and your {role} inbox is a firehose. Still — maybe join?",
    # More early
    "Credit card statement arrived. You're {age}, {debt_level}, and a {company_type} recruiter just pinged about a {role} gig with worse culture but +{salary}k.",
    "{family_situation}. Your mom needs help moving this weekend — same weekend as your {side_hustle} launch. You're {age} and feel torn in half.",
    "You're {age} in {city}. A mentor offers intros to {career_track} but wants unpaid 'prove it' work first. Rent is due. Do you grind for exposure?",
    "Partner wants a dog. Lease says no. You're {age}, {salary}k-ish income, {health_concern} acting up. Smuggle a pet or disappoint someone you love?",
    "You're {age}. Friend's wedding destination — expensive flight. Boss hints promotions go to people who skip vacations. {family_situation}.",
    # Mid phase (30s–50s)
    "Promotion on the table: more money, less sleep. You're {age}, {family_situation}, and {health_concern} came up at the last physical. Kids / partner / parents all need pieces of you.",
    "Kid's school called — behavioral stuff. Same week a {company_type} crisis needs you on-site in {city}. You're {age}. Spouse is furious you're always 'essential' at work.",
    "Mortgage, braces, aging parents. You're {age}, {debt_level} still hanging around, and a {role} peer just jumped to {risky_opportunity}. Feels like you're treading water.",
    "Doctor wants follow-up tests for {health_concern}. You're {age}, calendar packed, and work says 'can it wait until after Q4'.",
    "You're {age}. Teenager asks why you're never at games. Your manager scheduled a Saturday {side_hustle}-style workshop that's 'optional' but everyone knows it isn't.",
    "Spouse wants couples therapy. You're {age}, embarrassed about money ({debt_level}), and convinced you're too tired to unpack it.",
    "You're {age} in {city}. Parent fell; hospital text while you're in a {company_type} meeting. Do you leave mid-presentation?",
    "Career pivot whisper: leave {career_track} for something slower. You're {age}, {salary}k household, and terrified of starting over.",
    "You're {age}. Old friend died suddenly. You're rethinking everything — but bills don't care. {family_situation}.",
    "Boss offers {role} overseas assignment — 2 years. {family_situation}. Kids rooted in {city}. Money amazing. Roots not.",
    "You're {age}. Partner got a dream job in another state. Your {company_type} job is finally stable. Who sacrifices?",
    "Midlife checkup vibes: {health_concern}. Trainer costs money. You're {age}, time-poor, guilt-rich.",
    "You're {age}. Elder care costs just spiked. {debt_level}. Sibling says you always handle finances — you're exhausted.",
    "Conference trip could boost {career_track}. Same week as kid's recital. You're {age}. Spouse says choose wisely.",
    "You're {age}. Considering therapy for anxiety. Insurance is weird. Work will notice appointments. Still go?",
    "Refinance offer lands. Lower rate, longer trap. You're {age}, {family_situation}, {city} market weird.",
    "You're {age}. Volunteer role you love eats Saturdays. Family says you're avoiding home stress. Maybe true.",
    # Late phase (50s–70s)
    "You're {age}. Retirement calculators scream. {health_concern}. {family_situation}. Advisor says work two more years; body says maybe not.",
    "Doctor wants a procedure for {health_concern}. Recovery months. You're {age}, still working {role} part-time. Money vs longevity.",
    "You're {age}. Kids want you to move closer — {city} is expensive. Your friends are here. Downsizing means saying goodbye to the house full of memories.",
    "Legacy question: help grandkids' college now or keep buffer for {health_concern}. You're {age}, {debt_level} mostly gone, heart torn.",
    "You're {age}. Spouse retired; you're not ready. Tension at breakfast every day. {family_situation}.",
    "Old colleague offers consulting — good money, old stress. You're {age}, {health_concern}, thought you were done with {company_type} drama.",
    "You're {age} in {city}. Property tax jumped. Fixed income fiction. Consider moving somewhere cheaper — means leaving community.",
    "Aging parent needs daily help. You're {age}, your own {health_concern} flares. Sibling lives far and 'checks in' by text.",
    "You're {age}. Financial advisor pushes annuities. Feels like a sales pitch. {family_situation}. Trust vs paralysis.",
    "Dream trip vs roof replacement. You're {age}, want joy before knees give out, but {debt_level} whispers caution.",
    "You're {age}. Part-time {role} keeps you sharp but aches last days. Quit for hobbies?",
    "Class reunion — everyone comparing retirements. You're {age}, {salary}k wasn't enough to feel 'safe'. Pretend you're fine?",
    "You're {age}. Grandchild asks why you work still. Honest answer hurts. {family_situation}.",
    "Downsizing: donate, sell, or hoard? You're {age}, {health_concern}, attic is archaeology of your life.",
    "You're {age}. Long-term care insurance pitch at dinner. Premiums sting. Regret if you skip?",
]

assert len(SCENARIO_TEMPLATES) >= 40

PLACEHOLDERS: dict[str, list[str]] = {
    "city": ["Mumbai", "Bengaluru", "Delhi NCR", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"],
    "company_type": ["SaaS startup", "consulting firm", "bank", "hospital system", "retail chain", "public sector unit", "family business"],
    "role": ["engineering lead", "product manager", "sales director", "staff nurse", "accountant", "ops manager", "teacher", "field engineer"],
    "salary": ["42", "55", "68", "82", "95", "110", "38", "120"],
    "family_situation": [
        "Your partner is supportive but burned out",
        "You're single and mostly on your own",
        "You have aging parents who call daily",
        "You co-parent and custody weekends are sacred",
        "Your spouse travels for work half the month",
    ],
    "health_concern": ["borderline hypertension", "pre-diabetes", "chronic back pain", "sleep apnea", "anxiety spikes", "knee osteoarthritis", "heart murmur follow-up"],
    "side_hustle": ["tutoring", "freelance design", "delivery gigs", "content creation", "weekend coding contracts"],
    "relationship_move": ["move in together", "get engaged", "try long-distance", "start trying for kids", "buy a place jointly"],
    "debt_level": ["six figures of student loans", "a messy credit card stack", "a car loan you regret", "medical bills from last year", "a manageable but annoying EMI"],
    "risky_opportunity": ["a crypto-adjacent fund", "an angel round in a friend's company", "franchise outlet", "import-export scheme", "real estate flip"],
    "career_track": ["management", "IC technical path", "public policy", "medicine", "teaching", "finance"],
}


def _fill_template(tpl: str, state: BurenState) -> str:
    out = tpl
    keys = set(re.findall(r"\{(\w+)\}", tpl))
    for k in keys:
        choices = PLACEHOLDERS.get(k)
        if k == "age":
            val = str(state.age)
        elif choices:
            val = random.choice(choices)
        else:
            val = "something complicated"
        out = out.replace("{" + k + "}", val)
    return out


class ScenarioEngine:
    """Template-based scenarios with causal sampling and weakness targeting."""

    def __init__(
        self,
        rng: random.Random | None = None,
        bias_fn: Callable[[], str | None] | None = None,
    ):
        self._rng = rng or random.Random()
        self._bias_fn = bias_fn

    def causal_sample(self, state: BurenState) -> str:
        """Pick scenario based on current stats (causal chaining)."""
        bias = self._bias_fn() if self._bias_fn else None
        if bias:
            return self.generate_hard_scenario(bias, state)

        if state.health < 25:
            med = (
                "URGENT: you're {age}, {city}. {health_concern} is no longer abstract — "
                "specialist wants a decision this week. Work won't hold your {role} spot forever. {family_situation}."
            )
            return _fill_template(med, state)

        if state.wealth > 75 and state.health < 40:
            burn = (
                "You're {age}. Money finally feels okay ({company_type} stock, side deals) but your body is crashing — "
                "{health_concern}, no sleep, {family_situation}. Therapist says burnout; CFO of your life says grind."
            )
            return _fill_template(burn, state)

        if state.happiness < 30:
            iso = (
                "You're {age} in {city}. Room feels quiet wrong. {family_situation}. "
                "Friends drifted; {relationship_move} came up and you panicked. Work is the only place that still pings you."
            )
            return _fill_template(iso, state)

        if state.age > 55 and state.wealth < 40:
            ret = (
                "You're {age}. Retirement math is ugly — {debt_level}, {health_concern}, {family_situation}. "
                "A {role} gig wants more hours; body wants fewer. Every choice feels like closing a door."
            )
            return _fill_template(ret, state)

        # Otherwise random by phase
        phase = phase_from_age(state.age)
        if phase == "early":
            pool = SCENARIO_TEMPLATES[0:18]
        elif phase == "mid":
            pool = SCENARIO_TEMPLATES[18:36]
        else:
            pool = SCENARIO_TEMPLATES[36:]
        tpl = self._rng.choice(pool)
        return _fill_template(tpl, state)

    def generate_hard_scenario(self, weakness: str, state: BurenState | None = None) -> str:
        """Template-matching challenges for detected weakness (no LLM)."""
        w = weakness.lower()
        stub = state or BurenState(age=self._rng.randint(28, 58))
        if "wealth" in w or "wealth_bias" in w:
            tpl = (
                "You're {age}. A windfall tax mess, a sick parent, and a friend begging for a loan same week. "
                "{family_situation}. Every choice makes someone angry — and your spreadsheet says you're not as safe as you thought."
            )
        elif "health" in w or "health_neglect" in w:
            tpl = (
                "You're {age}. {health_concern} — doctor is blunt: change habits now or schedule scary tests. "
                "Work just put you on a {role} death-march deadline. {family_situation}."
            )
        elif "reason" in w or "shallow" in w:
            tpl = (
                "You're {age}. Two 'obvious' options in {city} both have hidden catches — visa issues, "
                "non-compete weirdness, {family_situation}. The easy answer is probably wrong."
            )
        elif "survival" in w or "failure" in w:
            tpl = (
                "You're {age}. Everything compounded — {debt_level}, {health_concern}, boss threatening PIP. "
                "{family_situation}. You need a plan that doesn't torch one pillar entirely."
            )
        else:
            tpl = self._rng.choice(SCENARIO_TEMPLATES)
        return _fill_template(tpl, stub)

