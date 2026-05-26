import json
from typing import Dict, List, Optional
from datetime import datetime


class TokenTracker:
    """
    Tracks API token usage to monitor costs and prevent budget exhaustion.
    """
    
    def __init__(self, budget_limit: Optional[int] = None):
        """
        Initialize token tracker.
        
        Args:
            budget_limit: Optional token budget limit (warning at 80%, error at 100%)
        """
        self.budget_limit = budget_limit
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.conversation_tokens = {}
        self.api_calls = []
        
    def track_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        conversation_id: Optional[int] = None,
        call_type: str = "text_generation"
    ):
        """
        Track token usage for an API call.
        
        Args:
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            conversation_id: Optional conversation ID
            call_type: Type of API call (text_generation, vision, etc.)
        """
        total = prompt_tokens + completion_tokens
        
        self.total_tokens += total
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        
        # Track per conversation
        if conversation_id is not None:
            if conversation_id not in self.conversation_tokens:
                self.conversation_tokens[conversation_id] = {
                    "total": 0,
                    "prompt": 0,
                    "completion": 0,
                    "calls": 0
                }
            
            self.conversation_tokens[conversation_id]["total"] += total
            self.conversation_tokens[conversation_id]["prompt"] += prompt_tokens
            self.conversation_tokens[conversation_id]["completion"] += completion_tokens
            self.conversation_tokens[conversation_id]["calls"] += 1
        
        # Log API call
        self.api_calls.append({
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "call_type": call_type,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total
        })
        
        # Check budget
        if self.budget_limit:
            usage_percent = (self.total_tokens / self.budget_limit) * 100
            
            if usage_percent >= 100:
                print(f"\n⚠️  BUDGET LIMIT REACHED: {self.total_tokens:,} / {self.budget_limit:,} tokens ({usage_percent:.1f}%)")
                print("   Consider stopping generation to avoid additional costs.")
            elif usage_percent >= 80:
                print(f"\n⚠️  Budget Warning: {self.total_tokens:,} / {self.budget_limit:,} tokens ({usage_percent:.1f}%)")
    
    def get_summary(self) -> Dict:
        """Get summary of token usage."""
        summary = {
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_api_calls": len(self.api_calls),
            "budget_limit": self.budget_limit,
            "budget_used_percent": (self.total_tokens / self.budget_limit * 100) if self.budget_limit else None,
            "conversations": self.conversation_tokens
        }
        
        return summary
    
    def print_summary(self):
        """Print token usage summary."""
        print("\n" + "=" * 60)
        print("Token Usage Summary")
        print("=" * 60)
        print(f"Total API calls: {len(self.api_calls)}")
        print(f"Total tokens: {self.total_tokens:,}")
        print(f"  - Prompt tokens: {self.total_prompt_tokens:,}")
        print(f"  - Completion tokens: {self.total_completion_tokens:,}")
        
        if self.budget_limit:
            usage_percent = (self.total_tokens / self.budget_limit) * 100
            remaining = self.budget_limit - self.total_tokens
            print(f"\nBudget: {self.total_tokens:,} / {self.budget_limit:,} tokens ({usage_percent:.1f}%)")
            print(f"Remaining: {remaining:,} tokens")
        
        if self.conversation_tokens:
            print(f"\nPer-conversation breakdown:")
            for conv_id, stats in sorted(self.conversation_tokens.items()):
                print(f"  Conversation {conv_id}: {stats['total']:,} tokens ({stats['calls']} calls)")
    
    def save(self, filepath: str):
        """Save token usage data to JSON file."""
        data = {
            "summary": self.get_summary(),
            "api_calls": self.api_calls
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Test
    tracker = TokenTracker(budget_limit=100000)
    
    tracker.track_usage(100, 50, conversation_id=1, call_type="vision")
    tracker.track_usage(200, 150, conversation_id=1, call_type="text_generation")
    tracker.track_usage(180, 120, conversation_id=2, call_type="text_generation")
    
    tracker.print_summary()
    tracker.save("test_token_usage.json")
    print("\n✓ Saved to test_token_usage.json")
