-- init_db.sql — reset the pdf2md database: drops all conversion data
-- (jobs, pages, images, events), the version row, and orphaned enum types.
-- Safe to re-run (every statement is IF EXISTS guarded).
--
--   psql -U postgres -h localhost -d pdf2md -f init_db.sql
--   venv\Scripts\python -m alembic upgrade head
--   venv\Scripts\python -m alembic current   (must print "b84dc46ed6a2 (head)")
--
-- Drops ALL conversion data (jobs, pages, images, events) plus the native
-- PG enum types (which plain table drops leave orphaned and which break a
-- later re-upgrade with 'type "jobstatus" already exists'). Safe to re-run:
-- every statement is IF EXISTS guarded.

DROP TABLE IF EXISTS events CASCADE;
DROP TABLE IF EXISTS images CASCADE;
DROP TABLE IF EXISTS pages CASCADE;
DROP TABLE IF EXISTS jobs CASCADE;
DROP TABLE IF EXISTS alembic_version CASCADE;

DROP TYPE IF EXISTS jobstatus;
DROP TYPE IF EXISTS pagestatus;
DROP TYPE IF EXISTS imagesource;
