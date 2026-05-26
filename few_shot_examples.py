"""
Few-shot examples for personas, strategies, and intents.
Used to improve dialogue generation quality through in-context learning.
"""

# ============================================================
# PERSONA EXAMPLES (8 personas × 2-3 examples each)
# ============================================================

PERSONA_EXAMPLES = {
    "budget-conscious": [
        {
            "context": "Agent mentions premium plan at ₹25,000",
            "response": "That's a bit steep for my budget. Are there any discounts available or perhaps a more basic plan that covers just the essentials?"
        },
        {
            "context": "Agent offers add-on for ₹3,000",
            "response": "I appreciate the offer, but I'm trying to keep costs down. What's the absolute minimum I need to be properly covered?"
        },
        {
            "context": "Agent explains comprehensive coverage benefits",
            "response": "I understand the value, but I need to watch every rupee. Can we start with third-party only and I'll upgrade later?"
        }
    ],
    
    "detail-oriented": [
        {
            "context": "Agent mentions zero depreciation coverage",
            "response": "Could you explain exactly what that covers? Does it include plastic and fiber parts as well, or just the metal components?"
        },
        {
            "context": "Agent says 'comprehensive coverage'",
            "response": "What specifically is included in 'comprehensive'? I'd like to see a detailed breakdown of all inclusions and exclusions."
        },
        {
            "context": "Agent mentions claim settlement ratio",
            "response": "What's the average claim settlement time? And are there any conditions that could delay or deny a claim?"
        }
    ],
    
    "skeptical": [
        {
            "context": "Agent claims fastest claim settlement",
            "response": "That sounds impressive, but could you share any customer reviews or actual data? I just want to ensure it's the right choice."
        },
        {
            "context": "Agent recommends specific insurer",
            "response": "Could you help me understand why this company specifically? I want to make sure I am getting the best deal on the market."
        },
        {
            "context": "Agent mentions award or certification",
            "response": "Awards can be bought. What's your actual claim settlement ratio, and can I verify that independently?"
        }
    ],
    
    "enthusiastic": [
        {
            "context": "Agent recommends comprehensive plan",
            "response": "That sounds perfect! I want the best protection for my new baby. What premium features can we add?"
        },
        {
            "context": "Agent mentions roadside assistance",
            "response": "Oh, that's fantastic! I love road trips. Does it cover the whole country? What else comes with this?"
        },
        {
            "context": "Agent explains zero depreciation",
            "response": "Brilliant! I definitely want that. My car is brand new and I want to keep it in showroom condition!"
        }
    ],
    
    "cautious": [
        {
            "context": "Agent suggests standard coverage",
            "response": "I'm worried that might not be enough. What if there's a major accident? Should I get higher coverage?"
        },
        {
            "context": "Agent mentions deductible amount",
            "response": "What happens if the damage is more than expected? I have young kids, so safety is my top priority."
        },
        {
            "context": "Agent explains policy limits",
            "response": "That makes me nervous. Is there an option with no limits or higher limits? I'd rather pay more for peace of mind."
        }
    ],
    
    "tech-savvy": [
        {
            "context": "Agent mentions claim process",
            "response": "Can I file claims through an app? I prefer handling everything digitally without paperwork."
        },
        {
            "context": "Agent explains policy features",
            "response": "Do you have AI-based claim assessment? I've heard some insurers use video inspection now."
        },
        {
            "context": "Agent recommends insurer",
            "response": "What's their digital infrastructure like? I want instant policy issuance and real-time claim tracking."
        }
    ],
    
    "first-time buyer": [
        {
            "context": "Agent mentions IDV",
            "response": "Pardon me, could you explain what IDV means? This is my first car and I'm still learning these terms."
        },
        {
            "context": "Agent explains comprehensive vs third-party",
            "response": "I'm a little confused. Could you tell me what most people choose for a new car? I'd appreciate your guidance."
        },
        {
            "context": "Agent mentions add-ons",
            "response": "There are quite a few options! Could you help me understand which ones are actually the most essential for a new driver?"
        }
    ],
    
    "experienced buyer": [
        {
            "context": "Agent starts explaining basics",
            "response": "Thanks, I am actually fairly familiar since I've had a few policies before. Could we skip straight to the IDV calculation and NCB transfer?"
        },
        {
            "context": "Agent recommends coverage",
            "response": "I appreciate the recommendation. I usually prefer comprehensive with zero dep, engine protection, and return-to-invoice. Could you give me your best quote?"
        },
        {
            "context": "Agent mentions claim process",
            "response": "Got it. Could you also confirm your average claim settlement time and if you have cashless options at XYZ garage?"
        }
    ]
}


# ============================================================
# STRATEGY EXAMPLES (6 strategies × 2-3 examples each)
# ============================================================

STRATEGY_EXAMPLES = {
    "Default": [
        {
            "context": "User asks general question about insurance",
            "response": "I'd be happy to help you find the right coverage for your new car. What's most important to you - comprehensive protection or keeping costs manageable?"
        },
        {
            "context": "User mentions they need insurance",
            "response": "Great! Let me guide you through the options. We have plans ranging from basic third-party coverage to comprehensive packages with add-ons."
        }
    ],
    
    "Credibility": [
        {
            "context": "User asks why they should trust the company",
            "response": "That's a fair question. At Aspire, we've settled over 98% of our motor claims last year alone. We've been recognized as the 'Most Trusted Insurer' in 2024, and our 24/7 support ensures you're never stranded."
        },
        {
            "context": "User questions claim settlement speed",
            "response": "We have one of the fastest settlement records in the industry - 7 days on average for cashless claims. Plus, we have 8,200+ network garages nationwide for your convenience."
        },
        {
            "context": "User is skeptical about promises",
            "response": "I understand your concern. Our 99% claim settlement ratio is verified by IRDAI. We also have a 4.5-star rating from over 50,000 customers on independent review platforms."
        }
    ],
    
    "Emotional": [
        {
            "context": "User says insurance feels like a waste",
            "response": "I understand. Think of it as buying peace of mind for your family. If something unexpected happens to your new car, you shouldn't have to worry about the financial burden while dealing with the stress."
        },
        {
            "context": "User mentions family safety",
            "response": "Your family's safety is priceless. With our comprehensive coverage, you can drive knowing that if anything happens, your loved ones are protected from financial hardship."
        },
        {
            "context": "User worries about accidents",
            "response": "That worry is exactly why we're here. Imagine the relief of knowing that even in the worst scenario, you're fully covered and won't face devastating repair bills."
        }
    ],
    
    "Logical": [
        {
            "context": "User asks about value for money",
            "response": "Let me break down the math. A ₹25,000 premium covers potential damages up to ₹8 lakhs. Even one major accident would cost you 3x the annual premium without insurance."
        },
        {
            "context": "User compares different plans",
            "response": "Here's the comparison: Plan A costs ₹18,000 with ₹5,000 deductible. Plan B costs ₹25,000 with zero deductible. If you claim twice a year, Plan B saves you ₹10,000 overall."
        },
        {
            "context": "User questions add-on value",
            "response": "Zero depreciation costs ₹2,000 extra but saves you 30-40% on claim payouts. For a ₹50,000 repair, you'd save ₹15,000-20,000. It pays for itself in one claim."
        }
    ],
    
    "Personal": [
        {
            "context": "User mentions they have an SUV",
            "response": "I see you've got a premium SUV there - excellent choice for family trips! Given its high value, even a small scratch could be expensive to fix. Our SUV-tier plan covers those minor cosmetic issues that standard plans often miss."
        },
        {
            "context": "User is worried about price for their vehicle",
            "response": "I completely understand your concern. For your specific model, we've designed a plan that balances comprehensive coverage with affordability. Many customers with similar vehicles choose this option."
        },
        {
            "context": "User mentions they drive long distances",
            "response": "Since you drive long distances, roadside assistance is crucial for you. Our plan includes unlimited towing, fuel delivery, and tire changes anywhere in India - perfect for highway travelers like yourself."
        }
    ],
    
    "Persona": [
        {
            "context": "User asks about claim experience",
            "response": "Let me share a recent example. Last month, a customer with a similar car had a fender bender. They filed through our app at 10 AM, got approval by 2 PM, and their car was fixed within 3 days - all cashless."
        },
        {
            "context": "User worries about hidden costs",
            "response": "I had a customer last week with the same concern. We went through every line item together, and they were relieved to see there were no surprises. What you see is what you pay - no hidden charges."
        },
        {
            "context": "User questions if add-ons are worth it",
            "response": "A colleague of mine skipped engine protection to save ₹1,500. Three months later, waterlogging damaged his engine - ₹80,000 repair. He learned the hard way. I always recommend it, especially during monsoon season."
        }
    ]
}


# ============================================================
# INTENT EXAMPLES (8 intents × 2-3 examples each)
# ============================================================

INTENT_EXAMPLES = {
    "I-1 Session Onset": [
        {
            "context": "Starting conversation",
            "response": "Hi, I just bought a new car and need to get it insured. Can you help me find a good policy?"
        },
        {
            "context": "Starting conversation",
            "response": "Hello, I'm looking for insurance for my brand new vehicle. What options do you have?"
        }
    ],
    
    "I-2 Coverage Inquiry": [
        {
            "context": "Agent mentions comprehensive plan",
            "response": "What exactly does comprehensive coverage include? Does it cover theft and natural disasters?"
        },
        {
            "context": "Agent recommends a policy",
            "response": "Can you explain what's covered under this plan? I want to understand the details before deciding."
        },
        {
            "context": "Following up on features",
            "response": "You mentioned cashless repairs - how does that work exactly?"
        }
    ],
    
    "I-3 Quote Solicitation": [
        {
            "context": "After hearing about coverage",
            "response": "That sounds good. What would the annual premium be for my car?"
        },
        {
            "context": "Comparing options",
            "response": "Can you give me a quote for both the basic and comprehensive plans?"
        },
        {
            "context": "Ready to see pricing",
            "response": "I'm interested. How much would this cost me per year?"
        }
    ],
    
    "I-4 Concern Disclosure": [
        {
            "context": "Agent mentions exclusions",
            "response": "I'm worried about what's not covered. What if I need repairs that aren't included?"
        },
        {
            "context": "Hearing about deductibles",
            "response": "That deductible seems high. What if I can't afford it when I need to make a claim?"
        },
        {
            "context": "Discussing claim process",
            "response": "I've heard horror stories about claims being rejected. How can I be sure mine will be approved?"
        }
    ],
    
    "I-5 Clarification Request": [
        {
            "context": "After initial explanation",
            "response": "Can you tell me more about the add-ons available? What are my options?"
        },
        {
            "context": "Following up on features",
            "response": "You mentioned roadside assistance - what exactly does that include?"
        },
        {
            "context": "Seeking clarification",
            "response": "I'd like to know more about the claim settlement process. How long does it typically take?"
        }
    ],
    
    "I-6 Premium Negotiation": [
        {
            "context": "After receiving quote",
            "response": "That's a bit more than I expected. Is there any room for negotiation or discounts available?"
        },
        {
            "context": "Comparing with other quotes",
            "response": "I got a lower quote from another company. Can you match or beat their price?"
        },
        {
            "context": "Budget constraints",
            "response": "I really want this coverage, but ₹25,000 is stretching my budget. Can you do any better?"
        }
    ],
    
    "I-7 Renewal Commitment": [
        {
            "context": "After successful negotiation",
            "response": "Alright, ₹23,000 works for me. What are the next steps to get this policy activated?"
        },
        {
            "context": "Satisfied with offer",
            "response": "Perfect! I'm happy with this plan. How do I proceed with the purchase?"
        },
        {
            "context": "Ready to commit",
            "response": "This sounds like exactly what I need. Let's go ahead with this policy."
        }
    ],
    
    "I-8 Offer Decline": [
        {
            "context": "Price too high",
            "response": "I appreciate your time, but ₹25,000 is just beyond my budget right now. I'll need to look at other options."
        },
        {
            "context": "Not convinced",
            "response": "Thanks for the information, but I'm not convinced this is the right fit for me. I'll think about it."
        },
        {
            "context": "Found better alternative",
            "response": "I've decided to go with another provider who offered a better deal. Thank you for your help though."
        }
    ]
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_persona_examples(persona: str, max_examples: int = 3) -> str:
    """
    Get formatted few-shot examples for a specific persona.
    
    Args:
        persona: Persona type (e.g., "budget-conscious")
        max_examples: Maximum number of examples to return
        
    Returns:
        Formatted string with examples
    """
    examples = PERSONA_EXAMPLES.get(persona, [])[:max_examples]
    
    if not examples:
        return ""
    
    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.append(f"Example {i}:\nContext: {ex['context']}\nYou: \"{ex['response']}\"")
    
    return "\n\n".join(formatted)


def get_strategy_examples(strategy: str, max_examples: int = 3) -> str:
    """
    Get formatted few-shot examples for a specific strategy.
    
    Args:
        strategy: Strategy type (e.g., "Credibility")
        max_examples: Maximum number of examples to return
        
    Returns:
        Formatted string with examples
    """
    examples = STRATEGY_EXAMPLES.get(strategy, [])[:max_examples]
    
    if not examples:
        return ""
    
    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.append(f"Example {i}:\nUser: \"{ex['context']}\"\nAgent: \"{ex['response']}\"")
    
    return "\n\n".join(formatted)


def get_intent_examples(intent: str, max_examples: int = 3) -> str:
    """
    Get formatted few-shot examples for a specific intent.
    
    Args:
        intent: Intent type (e.g., "I-3 Quote Solicitation")
        max_examples: Maximum number of examples to return
        
    Returns:
        Formatted string with examples
    """
    examples = INTENT_EXAMPLES.get(intent, [])[:max_examples]
    
    if not examples:
        return ""
    
    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.append(f"Example {i}:\nContext: {ex['context']}\nYou: \"{ex['response']}\"")
    
    return "\n\n".join(formatted)


if __name__ == "__main__":
    print("Testing Few-Shot Examples Module...\n")
    
    # Test persona examples
    print("=== PERSONA EXAMPLES ===")
    print(get_persona_examples("budget-conscious", 2))
    
    print("\n\n=== STRATEGY EXAMPLES ===")
    print(get_strategy_examples("Credibility", 2))
    
    print("\n\n=== INTENT EXAMPLES ===")
    print(get_intent_examples("I-3 Quote Solicitation", 2))
    
    print("\n\n✓ All examples loaded successfully!")
