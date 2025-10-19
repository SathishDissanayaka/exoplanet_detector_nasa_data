from config.supabase import supabase

def setup_database():
    """Sets up the required tables in Supabase"""
    
    # Create predictions table
    try:
        # Create predictions table using SQL
        supabase.table('predictions').select('*').limit(1).execute()
    except:
        supabase.query("""
            CREATE TABLE IF NOT EXISTS predictions (
                id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
                user_id UUID REFERENCES auth.users(id),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
                model_name TEXT,
                features JSONB,
                prediction TEXT,
                confidence FLOAT,
                saved_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
            );

            -- Set up RLS (Row Level Security)
            ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

            -- Create policy to allow users to see only their own predictions
            CREATE POLICY "Users can view own predictions"
                ON predictions FOR SELECT
                USING (auth.uid() = user_id);

            -- Create policy to allow users to insert their own predictions
            CREATE POLICY "Users can insert own predictions"
                ON predictions FOR INSERT
                WITH CHECK (auth.uid() = user_id);
        """).execute()

        print("Created predictions table")

    # Create model_metrics table for tracking model performance
    try:
        supabase.table('model_metrics').select('*').limit(1).execute()
    except:
        supabase.query("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
                model_name TEXT,
                accuracy FLOAT,
                precision FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                training_date TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
                metadata JSONB
            );

            -- Set up RLS
            ALTER TABLE model_metrics ENABLE ROW LEVEL SECURITY;

            -- Allow all authenticated users to view metrics
            CREATE POLICY "Authenticated users can view model metrics"
                ON model_metrics FOR SELECT
                USING (auth.role() = 'authenticated');

            -- Only allow specific roles to insert metrics (you can modify this as needed)
            CREATE POLICY "Only specific roles can insert model metrics"
                ON model_metrics FOR INSERT
                WITH CHECK (auth.role() = 'authenticated');
        """).execute()

        print("Created model_metrics table")