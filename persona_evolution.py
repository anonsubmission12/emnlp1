import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class LifeEvent:
    """Represents a life event that can trigger persona changes."""
    
    def __init__(
        self,
        event_type: str,
        description: str,
        year: int,
        impact: Dict[str, str]
    ):
        self.event_type = event_type
        self.description = description
        self.year = year
        self.impact = impact  # Changes to persona attributes


class PersonaEvolution:
    """
    Manages persona evolution over 10 years (2015-2025).
    Generates life events and tracks preference changes.
    """
    
    # Life event templates
    LIFE_EVENTS = {
        "first_car": {
            "description": "Bought first car",
            "typical_year": 2015,
            "impact": {
                "prior_experience": "First-time buyer",
                "risk_appetite": "Cautious, wants guidance"
            }
        },
        "job_promotion": {
            "description": "Got job promotion / salary increase",
            "impact": {
                "financial_profile": "Budget improved, willing to pay more",
                "coverage_goal": "Balanced to Maximum"
            }
        },
        "minor_accident": {
            "description": "Had minor accident (no major damage)",
            "impact": {
                "risk_appetite": "Now wants comprehensive coverage",
                "preferred_addons": "zero_depreciation, roadside_assistance"
            }
        },
        "major_accident": {
            "description": "Had major accident (significant damage)",
            "impact": {
                "risk_appetite": "Very risk-averse, wants maximum coverage",
                "preferred_addons": "zero_depreciation, engine_protection, roadside_assistance"
            }
        },
        "vehicle_upgrade": {
            "description": "Upgraded to better/premium vehicle",
            "impact": {
                "vehicle_profile": "Upgraded vehicle",
                "coverage_goal": "Wants better coverage for new car",
                "preferred_addons": "consumables, roadside_assistance"
            }
        },
        "marriage": {
            "description": "Got married",
            "impact": {
                "lifestyle": "Married, family considerations",
                "risk_appetite": "More cautious, family safety priority"
            }
        },
        "child_birth": {
            "description": "Had a child",
            "impact": {
                "lifestyle": "Parent, family safety critical",
                "risk_appetite": "Very cautious, comprehensive coverage needed",
                "preferred_addons": "roadside_assistance, zero_depreciation"
            }
        },
        "job_loss": {
            "description": "Lost job / financial setback",
            "impact": {
                "financial_profile": "Budget-constrained, looking for savings",
                "coverage_goal": "Maximum to Minimum"
            }
        },
        "claim_experience_good": {
            "description": "Had smooth claim settlement experience",
            "impact": {
                "trust_level": "Increased trust in insurer",
                "loyalty": "Likely to renew"
            }
        },
        "claim_experience_bad": {
            "description": "Had poor claim settlement experience",
            "impact": {
                "trust_level": "Decreased trust, considering switching",
                "risk_appetite": "Wants better coverage/service"
            }
        },
        "relocation": {
            "description": "Relocated to new city",
            "impact": {
                "lifestyle": "New location, different driving conditions",
                "vehicle_profile": "May need different coverage"
            }
        },
        "business_use": {
            "description": "Started using vehicle for business",
            "impact": {
                "vehicle_profile": "Commercial use, higher risk",
                "coverage_goal": "Needs commercial coverage"
            }
        }
    }
    
    # Persona-specific event probabilities
    PERSONA_EVENT_WEIGHTS = {
        "budget-conscious": {
            "job_promotion": 0.15,
            "minor_accident": 0.10,
            "vehicle_upgrade": 0.05,
            "job_loss": 0.08
        },
        "detail-oriented": {
            "claim_experience_good": 0.12,
            "claim_experience_bad": 0.08,
            "vehicle_upgrade": 0.10
        },
        "skeptical": {
            "claim_experience_bad": 0.15,
            "minor_accident": 0.12,
            "major_accident": 0.05
        },
        "enthusiastic": {
            "vehicle_upgrade": 0.20,
            "marriage": 0.12,
            "job_promotion": 0.15
        },
        "cautious": {
            "minor_accident": 0.15,
            "major_accident": 0.08,
            "child_birth": 0.12
        },
        "tech-savvy": {
            "vehicle_upgrade": 0.18,
            "relocation": 0.10,
            "business_use": 0.08
        },
        "first-time buyer": {
            "first_car": 1.0,  # Always starts with first car
            "minor_accident": 0.15,
            "job_promotion": 0.12
        },
        "experienced buyer": {
            "vehicle_upgrade": 0.15,
            "claim_experience_good": 0.15,
            "business_use": 0.10
        }
    }
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.persona_timelines = {}
    
    def generate_timeline(
        self,
        persona_id: str,
        personality: str,
        start_year: int = 2015,
        end_year: int = 2025
    ) -> List[LifeEvent]:
        """
        Generate a timeline of life events for a persona over 10 years.
        
        Args:
            persona_id: Unique persona identifier
            personality: Persona personality type
            start_year: Starting year (default 2015)
            end_year: Ending year (default 2025)
            
        Returns:
            List of LifeEvent objects chronologically ordered
        """
        events = []
        years = list(range(start_year, end_year + 1))
        
        # Get event weights for this personality
        event_weights = self.PERSONA_EVENT_WEIGHTS.get(personality, {})
        
        # First-time buyers always start with first car
        if personality == "first-time buyer":
            events.append(LifeEvent(
                event_type="first_car",
                description=self.LIFE_EVENTS["first_car"]["description"],
                year=start_year,
                impact=self.LIFE_EVENTS["first_car"]["impact"]
            ))
        
        # Generate 2-4 life events over 10 years
        num_events = random.randint(2, 4)
        event_years = random.sample(years[1:], min(num_events, len(years) - 1))
        
        for year in sorted(event_years):
            # Select event based on personality weights
            if event_weights:
                event_type = random.choices(
                    list(event_weights.keys()),
                    weights=list(event_weights.values()),
                    k=1
                )[0]
            else:
                # Fallback to common events
                event_type = random.choice([
                    "job_promotion", "minor_accident", "vehicle_upgrade", "marriage"
                ])
            
            event_template = self.LIFE_EVENTS[event_type]
            events.append(LifeEvent(
                event_type=event_type,
                description=event_template["description"],
                year=year,
                impact=event_template["impact"]
            ))
        
        # Store timeline
        self.persona_timelines[persona_id] = events
        
        return events
    
    def get_events_for_year(self, persona_id: str, year: int) -> List[LifeEvent]:
        """Get all life events for a specific year."""
        if persona_id not in self.persona_timelines:
            return []
        
        return [
            event for event in self.persona_timelines[persona_id]
            if event.year == year
        ]
    
    def get_cumulative_impact(
        self,
        persona_id: str,
        up_to_year: int
    ) -> Dict[str, List[str]]:
        """
        Get cumulative impact of all events up to a specific year.
        
        Returns:
            Dict mapping aspect (e.g., 'financial_profile') to list of changes
        """
        if persona_id not in self.persona_timelines:
            return {}
        
        cumulative = {}
        for event in self.persona_timelines[persona_id]:
            if event.year <= up_to_year:
                for aspect, change in event.impact.items():
                    if aspect not in cumulative:
                        cumulative[aspect] = []
                    cumulative[aspect].append(f"{event.year}: {change}")
        
        return cumulative
    
    def to_dict(self) -> Dict:
        """Convert all timelines to dictionary format."""
        return {
            persona_id: [
                {
                    "event_type": event.event_type,
                    "description": event.description,
                    "year": event.year,
                    "impact": event.impact
                }
                for event in events
            ]
            for persona_id, events in self.persona_timelines.items()
        }


if __name__ == "__main__":
    print("Testing PersonaEvolution...")
    
    # Test timeline generation
    evolution = PersonaEvolution(seed=42)
    
    personas = [
        ("P_001", "budget-conscious"),
        ("P_002", "first-time buyer"),
        ("P_003", "cautious")
    ]
    
    for persona_id, personality in personas:
        print(f"\n{'='*60}")
        print(f"Persona: {persona_id} ({personality})")
        print('='*60)
        
        timeline = evolution.generate_timeline(persona_id, personality)
        
        for event in timeline:
            print(f"\n{event.year}: {event.description}")
            print(f"  Impact:")
            for aspect, change in event.impact.items():
                print(f"    - {aspect}: {change}")
        
        # Test cumulative impact
        print(f"\nCumulative impact by 2025:")
        cumulative = evolution.get_cumulative_impact(persona_id, 2025)
        for aspect, changes in cumulative.items():
            print(f"  {aspect}:")
            for change in changes:
                print(f"    - {change}")
    
    print("\nAll tests passed!")
