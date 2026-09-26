-- =============================================================================
-- Snowflake Setup for Training `Concealed` on a GPU Compute Pool
-- =============================================================================

-- 1. Create Database, Schema, and Internal Stages for Images & Trained Models
CREATE DATABASE IF NOT EXISTS CONCEALED_DB;
USE DATABASE CONCEALED_DB;
CREATE SCHEMA IF NOT EXISTS PUBLIC;

-- Stage for your dataset of a couple thousand images (DIRECTORY = TRUE enables browsing)
CREATE STAGE IF NOT EXISTS IMAGE_STAGE
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  DIRECTORY = (ENABLE = TRUE);

-- Stage to store trained .pt checkpoints, .onnx exports, and visual sample grids
CREATE STAGE IF NOT EXISTS MODEL_STAGE
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  DIRECTORY = (ENABLE = TRUE);

-- 2. Create a GPU Compute Pool
--    - GPU_NV_S: 1x NVIDIA A10G (24 GB VRAM) -> Ideal for CLIP + SigLIP + DINOv2 ensemble
--    - GPU_NV_M: 4x NVIDIA A10G (96 GB VRAM)
CREATE COMPUTE POOL IF NOT EXISTS CONCEALED_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_S
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 600;

-- 3. Allow HuggingFace Hub & PyPI egress so the container can pull open-source ViT weights
CREATE OR REPLACE NETWORK RULE HF_PYPI_NETWORK_RULE
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = (
    'huggingface.co:443',
    'cdn-lfs.huggingface.co:443',
    'cdn-lfs-us-1.huggingface.co:443',
    'cas-bridge.xethub.hf.co:443',
    'pypi.org:443',
    'files.pythonhosted.org:443'
  );

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION CONCEALED_HF_ACCESS
  ALLOWED_NETWORK_RULES = (HF_PYPI_NETWORK_RULE)
  ENABLED = TRUE;
