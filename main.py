import json
import os
import sys
import requests
from pathlib import Path
from datetime import datetime, timedelta
import random
from tqdm import tqdm
from dialogue_generator import DialogueGenerator
from graph import create_intent_graph, create_strategy_graph
from token_tracker import TokenTracker
from persona_evolution import PersonaEvolution

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# ============================================================
# CONFIGURATION
# ============================================================
TEST_MODE = False  # Set to False to generate all personas
TEST_PERSONAS = 3  # Number of personas in test mode
TOKEN_BUDGET = 500000  # Token budget limit (warning at 80%)
YEARS = list(range(2015, 2026))  # 2015-2025 (11 years)
CONVERSATIONS_PER_YEAR = (1, 1)  # Exactly 1 conversation per year (Max 11 per persona)

# Personalities to generate (in order). "budget-conscious" already done.
# Order: enthusiastic -> tech-savvy -> skeptical -> detail-oriented -> cautious -> first-time buyer
REMAINING_PERSONAS = [
    "enthusiastic",
    "tech-savvy",
    "skeptical",
    "detail-oriented",
    "cautious",
    "first-time buyer"
]

# budget-conscious used P_001-P_015 (5 traits x 3 incomes = 15 combos)
# Next persona starts at P_016
PERSONA_OFFSET = 15
# ============================================================

def validate_urls(image_links):
    """Checks each URL, removes inaccessible ones, saves them to a log."""
    print(f"\n[Validation] Checking {len(image_links)} URLs...")
    valid_urls = []
    inaccessible = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0'
    }
    for item in tqdm(image_links, desc="Validating URLs"):
        url = item.get("src")
        try:
            resp = requests.head(url, timeout=5, headers=headers)
            if resp.status_code == 405: # Method not allowed, try GET
                resp = requests.get(url, stream=True, timeout=5, headers=headers)
            if resp.status_code == 200:
                valid_urls.append(item)
            else:
                inaccessible.append(url)
        except Exception:
            inaccessible.append(url)
            
    if inaccessible:
        print(f"  Found {len(inaccessible)} inaccessible URLs.")
        with open("inaccessible_urls.txt", "w", encoding="utf-8") as f:
            f.write("Inaccessible URLs Log\n====================\n\n")
            f.write("\n".join(inaccessible))
    
    if not valid_urls:
         print("  CRITICAL: No valid URLs found! Using fallback.")
         valid_urls = [{"src": "fallback_image.jpg"}] # Mock fallback
         
    return valid_urls

def generate_random_timestamp(year=2025):
    """
    Generate a random timestamp within the specified year.
    
    Args:
        year: Year for the timestamp (default: 2025)
        
    Returns:
        ISO format timestamp string
    """
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23, 59, 59)
    
    time_between = end_date - start_date
    random_seconds = random.randint(0, int(time_between.total_seconds()))
    random_date = start_date + timedelta(seconds=random_seconds)
    
    return random_date.isoformat()


def main():
    """
    Main entry point for 10-year long-horizon dialogue generation system.
    Generates conversations spanning 2015-2025 with persistent persona memory.
    """
    print("=" * 60)
    print("10-Year Long-Horizon Dialogue Generation System")
    print("=" * 60)
    
    if TEST_MODE:
        print(f"\nTEST MODE ENABLED - Generating only {TEST_PERSONAS} persona(s)")
        print("   Set TEST_MODE = False in main.py to generate all personas")
    print()
    
    # Load insurance data
    print("\n[1/7] Loading insurance data...")
    try:
        with open("motor-insurance-updated.json", "r", encoding="utf-8") as f:
            insurance_data = json.load(f)
        print("Loaded motor-insurance-updated.json")
    except FileNotFoundError:
        print("Error: 'motor-insurance-updated.json' not found.")
        return
    
    # Load image URLs
    print("\n[2/7] Loading and validating vehicle image URLs...")
    try:
        with open("urls.json", "r", encoding="utf-8") as f:
            image_links = json.load(f)
        image_links = validate_urls(image_links)
        print(f"Loaded {len(image_links)} valid image URLs")
    except FileNotFoundError:
        print("Error: 'urls.json' not found.")
        return
    
    # Create output directories
    print("\n[3/7] Creating output directories...")
    Path("output").mkdir(exist_ok=True)
    Path("memory").mkdir(exist_ok=True)
    Path("memory/personas").mkdir(exist_ok=True)
    print("Created output/, memory/, and memory/personas/ directories")
    
    # Load existing output data to append to (budget-conscious already generated)
    print("\n[3.5/7] Loading existing output data to resume...")
    existing_conversations = []
    existing_thinking = []
    existing_memory_snapshots = []
    existing_persona_timelines = []
    
    if Path("output/conversations.json").exists():
        with open("output/conversations.json", "r", encoding="utf-8") as f:
            existing_conversations = json.load(f)
        print(f"  Loaded {len(existing_conversations)} existing conversations")
    
    if Path("output/thinking.json").exists():
        with open("output/thinking.json", "r", encoding="utf-8") as f:
            existing_thinking = json.load(f)
        print(f"  Loaded {len(existing_thinking)} existing thinking logs")
    
    if Path("output/memory_snapshots.json").exists():
        with open("output/memory_snapshots.json", "r", encoding="utf-8") as f:
            existing_memory_snapshots = json.load(f)
        print(f"  Loaded {len(existing_memory_snapshots)} existing memory snapshots")
    
    if Path("output/persona_timelines.json").exists():
        with open("output/persona_timelines.json", "r", encoding="utf-8") as f:
            existing_persona_timelines = json.load(f)
        print(f"  Loaded {len(existing_persona_timelines)} existing persona timelines")
    
    # Find starting conversation_id from existing data
    starting_conversation_id = 1
    if existing_conversations:
        max_existing_id = max(c["conversation_id"] for c in existing_conversations)
        starting_conversation_id = max_existing_id + 1
        print(f"  Resuming from conversation_id: {starting_conversation_id}")
    
    # Initialize generator
    print("\n[4/7] Initializing dialogue generator...")
    
    # Initialize token tracker
    token_tracker = TokenTracker(budget_limit=TOKEN_BUDGET)
    
    generator = DialogueGenerator(insurance_data, token_tracker=token_tracker)
    
    # Load or initialize global graph weights
    intent_graph_path = "memory/global_intent_graph.json"
    strategy_graph_path = "memory/global_strategy_graph.json"
    
    if generator.intent_graph.load_global_weights(intent_graph_path):
        print(f"Loaded cumulative intent graph weights from {intent_graph_path}")
    else:
        print("Initialized fresh intent graph weights")
    
    if generator.strategy_graph.load_global_weights(strategy_graph_path):
        print(f"Loaded cumulative strategy graph weights from {strategy_graph_path}")
    else:
        print("Initialized fresh strategy graph weights")
    
    print("Generator initialized with:")
    print(f"  - 8 user personas: {', '.join(generator.user_personalities)}")
    print(f"  - Intent graph with {len(generator.intent_graph.nodes)} nodes")
    print(f"  - Strategy graph with {len(generator.strategy_graph.nodes)} nodes")
    print(f"  - Token budget: {TOKEN_BUDGET:,} tokens")
    
    # Initialize persona evolution system
    print("\n[5/7] Initializing persona evolution system...")
    persona_evolution = PersonaEvolution(seed=42)
    
    # Configuration
    if TEST_MODE:
        base_personas = ["tech-savvy"]
        big_five_traits = ["conscientiousness"]
        inc_combs = ["Low"]
        print(f"Test Mode: Selected specific persona(s): {', '.join(base_personas)}")
    else:
        # Use REMAINING_PERSONAS (ordered list, excluding already-generated 'budget-conscious')
        base_personas = REMAINING_PERSONAS
        big_five_traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        inc_combs = ["Low", "Mid", "High"]
        print(f"Generating for remaining personas: {', '.join(base_personas)}")
        
    # Generate the cartesian product: 7 * 5 * 3 = 105 combinations
    personas_to_generate = []
    for bp in base_personas:
        for b5 in big_five_traits:
            for ir in inc_combs:
                personas_to_generate.append({
                    "personality": bp,
                    "dominant_trait": b5,
                    "income_range": ir
                })
    
    print(f"\nGeneration Configuration:")
    print(f"  - Remaining Personas to generate: {len(personas_to_generate)}")
    print(f"  - Persona IDs: P_{PERSONA_OFFSET + 1:03d} to P_{PERSONA_OFFSET + len(personas_to_generate):03d}")
    print(f"  - Years: {YEARS[0]}-{YEARS[-1]} ({len(YEARS)} years)")
    print(f"  - Conversations per year: {CONVERSATIONS_PER_YEAR[0]}-{CONVERSATIONS_PER_YEAR[1]}")
    print(f"  - Estimated new conversations: {len(personas_to_generate) * len(YEARS) * sum(CONVERSATIONS_PER_YEAR) // 2}")
    print(f"  - Existing conversations loaded: {len(existing_conversations)}")
    print(f"  - Starting conversation_id: {starting_conversation_id}")
    
    # Storage for all data - start with existing data
    all_conversations = list(existing_conversations)
    all_thinking = list(existing_thinking)
    all_memory_snapshots = list(existing_memory_snapshots)
    all_persona_timelines = list(existing_persona_timelines)
    
    # Generate conversations
    print(f"\n[6/7] Generating conversations...")
    conversation_id = starting_conversation_id
    
    for persona_idx, persona_dict in enumerate(personas_to_generate):
        # Offset persona numbering to continue after budget-conscious (P_001-P_015)
        actual_persona_num = persona_idx + 1 + PERSONA_OFFSET
        persona_id = f"P_{actual_persona_num:03d}"
        personality = persona_dict["personality"]
        dominant_trait = persona_dict["dominant_trait"]
        income_range = persona_dict["income_range"]
        
        print(f"\n{'='*60}")
        print(f"Persona {persona_idx + 1}/{len(personas_to_generate)}: {persona_id} ({personality} - High {dominant_trait.capitalize()} - {income_range})")
        print(f"{'='*60}")
        
        # Generate life events timeline for this persona
        life_events = persona_evolution.generate_timeline(persona_id, personality)
        print(f"Generated {len(life_events)} life events over {len(YEARS)} years")
        
        # Initialize persona memory (load if exists)
        persona_memory_path = f"memory/personas/{persona_id}.json"
        
        # Track persona timeline
        persona_timeline = {
            "persona_id": persona_id,
            "personality": personality,
            "dominant_trait": dominant_trait,
            "income_range": income_range,
            "life_events": [
                {
                    "year": event.year,
                    "event_type": event.event_type,
                    "description": event.description
                }
                for event in life_events
            ],
            "conversations": []
        }
        
        # Generate conversations for each year
        for year in YEARS:
            # Determine number of conversations this year
            num_convs_this_year = random.randint(*CONVERSATIONS_PER_YEAR)
            
            for conv_num in range(num_convs_this_year):
                # Get life events for this year
                year_events = persona_evolution.get_events_for_year(persona_id, year)
                
                # Get image URL (cycle through available images)
                image_url = image_links[conversation_id % len(image_links)]['src']
                
                try:
                    # Temporarily set personality for this conversation
                    original_personalities = generator.user_personalities
                    generator.user_personalities = [personality]
                    
                    # Generate conversation with persona memory
                    conversation_data, thinking, memory_snapshots = generator.generate_conversation(
                        image_url,
                        conversation_id,
                        target_turns=30,
                        persona_id=persona_id,
                        year=year,
                        life_events=year_events,
                        dominant_trait=dominant_trait,
                        income_range=income_range
                    )
                    
                    # Restore personalities
                    generator.user_personalities = original_personalities
                    
                    # Add timestamp and year
                    conversation_data['timestamp'] = generate_random_timestamp(year=year)
                    conversation_data['year'] = year
                    conversation_data['persona_id'] = persona_id
                    
                    # Store data
                    all_conversations.append(conversation_data)
                    all_thinking.extend(thinking)
                    all_memory_snapshots.extend(memory_snapshots)
                    
                    # Add to persona timeline
                    persona_timeline["conversations"].append({
                        "year": year,
                        "conversation_id": conversation_id,
                        "outcome": conversation_data.get("outcome", "Unknown"),
                        "turns": len(conversation_data.get("conversation", []))
                    })
                    
                    # Print progress
                    print(f"  {year}: Conv {conversation_id} - {len(conversation_data.get('conversation', []))} turns, Outcome: {conversation_data.get('outcome', 'Unknown')}")
                    
                    # Incremental save to prevent data loss on interruption
                    with open("output/conversations.json", "w", encoding="utf-8") as f:
                        json.dump(sorted(all_conversations, key=lambda x: x.get('timestamp', '')), f, indent=2, ensure_ascii=False)
                    
                    with open("output/thinking.json", "w", encoding="utf-8") as f:
                        json.dump(all_thinking, f, indent=2, ensure_ascii=False)
                    
                    with open("output/memory_snapshots.json", "w", encoding="utf-8") as f:
                        json.dump(all_memory_snapshots, f, indent=2, ensure_ascii=False)
                    
                    conversation_id += 1
                
                except Exception as e:
                    print(f"\nError in conversation {conversation_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    conversation_id += 1
                    continue
        
        # Save persona timeline (incremental save)
        all_persona_timelines.append(persona_timeline)
        with open("output/persona_timelines.json", "w", encoding="utf-8") as f:
            json.dump(all_persona_timelines, f, indent=2, ensure_ascii=False)
    
    # Save outputs
    print(f"\n[7/7] Saving outputs...")
    
    # Sort conversations by timestamp (chronological order)
    all_conversations.sort(key=lambda x: x.get('timestamp', ''))
    
    # Save conversations
    with open("output/conversations.json", "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_conversations)} conversations to output/conversations.json")
    
    # Save thinking logs
    with open("output/thinking.json", "w", encoding="utf-8") as f:
        json.dump(all_thinking, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_thinking)} thinking logs to output/thinking.json")
    
    # Save memory snapshots
    with open("output/memory_snapshots.json", "w", encoding="utf-8") as f:
        json.dump(all_memory_snapshots, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_memory_snapshots)} memory snapshots to output/memory_snapshots.json")
    
    # Save persona timelines
    with open("output/persona_timelines.json", "w", encoding="utf-8") as f:
        json.dump(all_persona_timelines, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_persona_timelines)} persona timelines to output/persona_timelines.json")
    
    # Save global graph weights
    generator.intent_graph.save_global_weights(intent_graph_path)
    generator.strategy_graph.save_global_weights(strategy_graph_path)
    print(f"Saved cumulative graph weights to memory/")
    
    # Save token usage
    token_summary = token_tracker.get_summary()
    with open("output/token_usage.json", "w", encoding="utf-8") as f:
        json.dump(token_summary, f, indent=2)
    print(f"Saved token usage to output/token_usage.json")
    
    # Final summary
    print(f"\n{'='*60}")
    print("Generation Complete!")
    print(f"{'='*60}")
    print(f"Total conversations: {len(all_conversations)}")
    print(f"Total personas: {len(all_persona_timelines)}")
    print(f"Years covered: {YEARS[0]}-{YEARS[-1]}")
    print(f"Total tokens used: {token_tracker.total_tokens:,}")
    print(f"Budget remaining: {token_tracker.budget_limit - token_tracker.total_tokens:,}")
    
    # Show sample persona timeline
    if all_persona_timelines:
        sample = all_persona_timelines[0]
        print(f"\nSample Persona Timeline ({sample['persona_id']}):")
        for conv in sample['conversations'][:5]:
            print(f"  {conv['year']}: Conv {conv['conversation_id']} - {conv['turns']} turns, {conv['outcome']}")
        if len(sample['conversations']) > 5:
            print(f"  ... and {len(sample['conversations']) - 5} more conversations")


if __name__ == "__main__":
    main()
