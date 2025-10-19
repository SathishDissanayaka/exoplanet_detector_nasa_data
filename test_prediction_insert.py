"""
Test script to diagnose Supabase predictions table insertion issues
"""
from config.supabase import supabase
import traceback

def test_basic_insert():
    """Test a basic insert without user authentication"""
    print("=" * 60)
    print("TEST 1: Basic Insert (No User)")
    print("=" * 60)
    try:
        data = {
            'model_name': 'test_model',
            'prediction': 'CONFIRMED',
            'confidence': 0.95
        }
        print(f"Attempting to insert: {data}")
        response = supabase.table('predictions').insert(data).execute()
        print("✅ SUCCESS!")
        print(f"Response: {response}")
    except Exception as e:
        print("❌ FAILED!")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
    print()


def test_insert_with_features():
    """Test insert with features field"""
    print("=" * 60)
    print("TEST 2: Insert with Features (JSONB)")
    print("=" * 60)
    try:
        data = {
            'model_name': 'test_model',
            'prediction': 'CANDIDATE',
            'confidence': 0.75,
            'features': {
                'koi_prad': 1.5,
                'koi_teq': 300.0,
                'ra': 123.45,
                'dec': 67.89
            }
        }
        print(f"Attempting to insert: {data}")
        response = supabase.table('predictions').insert(data).execute()
        print("✅ SUCCESS!")
        print(f"Response: {response}")
    except Exception as e:
        print("❌ FAILED!")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
    print()


def test_select():
    """Test if we can read from the table"""
    print("=" * 60)
    print("TEST 3: Select from Table")
    print("=" * 60)
    try:
        response = supabase.table('predictions').select("*").limit(5).execute()
        print("✅ SUCCESS!")
        print(f"Found {len(response.data)} records")
        if response.data:
            print("Sample record:")
            print(response.data[0])
    except Exception as e:
        print("❌ FAILED!")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
    print()


def test_table_exists():
    """Test if the predictions table exists"""
    print("=" * 60)
    print("TEST 4: Check if Table Exists")
    print("=" * 60)
    try:
        # Try to get the table structure by doing a select with limit 0
        response = supabase.table('predictions').select("*").limit(0).execute()
        print("✅ Table exists!")
    except Exception as e:
        print("❌ Table might not exist or there's a permission issue")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
    print()


def test_rls_policies():
    """Test RLS policies"""
    print("=" * 60)
    print("TEST 5: Row Level Security Check")
    print("=" * 60)
    print("NOTE: If inserts fail but selects work, RLS policies might be blocking inserts")
    print("You may need to:")
    print("1. Disable RLS temporarily: ALTER TABLE predictions DISABLE ROW LEVEL SECURITY;")
    print("2. Or add a policy that allows service_role inserts")
    print("3. Or ensure the SUPABASE_KEY is the service_role key (not anon key)")
    print()


if __name__ == "__main__":
    print("\n🔍 SUPABASE PREDICTIONS TABLE DIAGNOSTIC TEST\n")
    
    test_table_exists()
    test_select()
    test_basic_insert()
    test_insert_with_features()
    test_rls_policies()
    
    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\n📋 Common Issues & Solutions:\n")
    print("1. RLS Policy Blocking:")
    print("   - Check if your SUPABASE_KEY is the service_role key (has full access)")
    print("   - Or modify RLS policies to allow your operations")
    print()
    print("2. UUID Format Issues:")
    print("   - user_id must be a valid UUID string")
    print("   - Ensure it references an existing user in auth.users")
    print()
    print("3. Column Mismatch:")
    print("   - Verify table columns match the data structure")
    print("   - Check if features column is properly defined as JSONB")
    print()
    print("4. Check Supabase Dashboard:")
    print("   - Go to Table Editor to see if table exists")
    print("   - Go to Authentication > Policies to check RLS policies")
    print("   - Go to Settings > API to verify you're using the right key")
    print()
