"""Fixed label sets shared by topic labeling (s20) and document labeling (s50)."""

INDUSTRIES = {
    "software_it": "Software, cloud & IT services (incl. AI model developers)",
    "semiconductors_hardware": "Semiconductors, chips, devices & data-center hardware",
    "telecom": "Telecommunications & networks",
    "finance_insurance": "Banking, investing, payments & insurance",
    "healthcare_life_sciences": "Healthcare providers, pharma, biotech & medical devices",
    "education": "Schools, universities & education services",
    "media_entertainment": "News, publishing, film, music, gaming & advertising",
    "retail_consumer": "Retail, e-commerce & consumer goods",
    "manufacturing_industrial": "Manufacturing, industrial automation & robotics",
    "transportation_automotive": "Automotive, logistics, aviation & transportation",
    "energy_utilities": "Energy, utilities & natural resources",
    "legal_professional": "Legal, accounting, consulting & HR services",
    "government_defense": "Government, public services, defense & security",
    "agriculture_food": "Agriculture & food production",
    "real_estate_construction": "Real estate & construction",
}
NO_INDUSTRY = "none"

# Industries that build or sell AI itself, as opposed to sectors adopting it.
SUPPLY_SIDE = {"software_it", "semiconductors_hardware"}

MECHANISMS = {
    "automation": "AI performs tasks previously done by people",
    "augmentation": "AI assists people who still do the work",
    "cost_efficiency": "AI lowers cost or raises throughput of existing processes",
    "new_offerings": "AI enables new products, services or revenue lines",
    "workflow_redesign": "Organizations restructure processes, roles or operating models around AI",
}

DIRECTIONS = ["positive", "negative", "mixed", "neutral"]

OUTCOMES = {
    "deployed": "an organization other than the AI vendor uses AI in its operations",
    "pilot_or_plan": "an organization is piloting, planning or budgeting AI adoption",
    "setback": "an adoption failed, was paused, rolled back, banned or caused harm",
    "product_launch": "a vendor releases or announces an AI product, with no specific adopter",
    "none": "no specific adoption or product is described",
}

FACTORS = {
    "data": "data availability, quality or privacy",
    "cost_roi": "cost, pricing or evidence of return on investment",
    "regulation_policy": "laws, regulation, lawsuits or government policy",
    "talent_skills": "workforce skills, hiring or training",
    "integration": "integration with existing systems and processes",
    "trust_safety": "accuracy, hallucination, bias, security or public trust",
    "compute_infrastructure": "compute, chips, energy or infrastructure capacity",
    "partnerships": "vendors, partnerships or ecosystem support",
    "leadership_strategy": "executive commitment, strategy or organizational readiness",
}
FACTOR_ROLES = ["enabler", "barrier"]


def industry_menu() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in INDUSTRIES.items()) + f"\n- {NO_INDUSTRY}: no specific industry"


def menu(d: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in d.items())
