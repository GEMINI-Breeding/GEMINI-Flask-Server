-- Initialize PostGIS extension for GEMINI database
-- This script runs automatically on first database startup

-- Enable PostGIS spatial database extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Enable PostGIS topology extension (optional but useful)
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Enable PostGIS raster extension (for raster/DEM support)
CREATE EXTENSION IF NOT EXISTS postgis_raster;

-- Verify PostGIS is installed
DO $$
BEGIN
    RAISE NOTICE 'PostGIS version: %', PostGIS_version();
END $$;
