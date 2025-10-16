"""
Simple Multi-User Memoria Example
Basic demonstration of creating separate memories for different users using namespaces

Configure the model by editing ``agents.default_model`` in ``memoria.json``.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion

from memoria import Memoria

CONFIG_PATH = Path(__file__).resolve().parent / "memoria.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).resolve().parents[1] / "memoria.json"
with open(CONFIG_PATH) as f:
    MODEL = json.load(f)["agents"]["default_model"]

load_dotenv()

# Global database for all users
DATABASE_PATH = "sqlite:///multiuser_memory.db"


def create_user_memory(user_id: str) -> Memoria:
    """Create a Memoria instance for a specific user using unique namespace"""
    print(f"👤 Creating memory for user: {user_id}")

    # Each user gets their own namespace in the shared database
    user_memory = Memoria(
        database_connect=DATABASE_PATH,
        namespace=f"user_{user_id}",
        conscious_ingest=True,
    )

    user_memory.enable()

    print(f"✅ Memory enabled for {user_id} with namespace: user_{user_id}")
    return user_memory


def user_chat(user_id: str, message: str) -> str:
    """Handle a chat message from a specific user"""
    print(f"\n💬 {user_id}: {message}")

    # The memory context is automatically handled by the user's namespace
    response = completion(
        model=MODEL, messages=[{"role": "user", "content": message}]
    )

    answer = response.choices[0].message.content
    print(f"🤖 Assistant to {user_id}: {answer}")
    return answer


def main():
    """Simple multi-user demonstration with shared database"""
    print("👥 Simple Multi-User Memoria Demo (Shared Database)")
    print("=" * 50)
    print(f"🗄️  Using shared database: {DATABASE_PATH}")
    print("🏷️  Each user gets isolated namespace")

    # Create memory for different users
    create_user_memory("alice")
    create_user_memory("bob")

    print("\n📝 User Conversations:")
    print("-" * 20)

    # Alice introduces herself
    user_chat(
        "alice", "Hi! I'm Alice, a software engineer working on web applications."
    )

    # Bob introduces himself
    user_chat("bob", "Hello! I'm Bob, a student studying machine learning.")

    # Alice shares preferences
    user_chat("alice", "I prefer using React and Node.js for my projects.")

    # Bob shares interests
    user_chat(
        "bob", "I'm particularly interested in neural networks and deep learning."
    )

    # Test memory isolation - each user should only remember their own info
    print("\n🧠 Testing Memory Isolation:")
    print("-" * 30)

    # Alice asks about her preferences
    user_chat("alice", "What do you remember about my technical preferences?")

    # Bob asks about his interests
    user_chat("bob", "What subjects am I studying?")

    # Cross-user test - Alice shouldn't know about Bob
    user_chat("alice", "Do you know anything about Bob or machine learning students?")

    # Cross-user test - Bob shouldn't know about Alice
    user_chat("bob", "Do you know about any web developers named Alice?")

    # Add a new user dynamically
    print("\n➕ Adding new user dynamically:")
    user_chat("charlie", "I'm Charlie, and I love cooking Italian food!")
    user_chat("charlie", "What do you remember about my interests?")

    print("\n✅ Demo completed!")
    print("📋 Summary:")
    print("  ✅ Single shared database for all users")
    print("  ✅ Isolated namespaces per user (user_alice, user_bob, user_charlie)")
    print("  ✅ Memory isolation maintained between users")
    print("  ✅ Efficient resource usage with shared database")
    print(f"  💾 Database file: {DATABASE_PATH}")


if __name__ == "__main__":
    main()
