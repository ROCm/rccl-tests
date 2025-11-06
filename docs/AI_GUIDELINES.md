# AI Assistant Guidelines for RCCL Testing Project

## General Working Principles

### Change Management
- **When I say something should work** and you determine it needs more than a couple of minor scripting changes to fix, **stop and ask for help** rather than continuing to make modifications
- Avoid making source code changes unless explicitly approved
- Prioritize understanding and working with existing tools over creating new ones

### Project Focus
- Focus on performance analysis scripts in `/work/meadows/rccl/scripts`
- Ensure mpirun is accessible before running any benchmarks
- Avoid running benchmarks unless specifically requested
- Rebuild as needed but do not make source changes

### Communication Style
- Be proactive about asking for clarification when tasks become complex
- Reference this guidelines file in conversations to maintain consistency
- Stop and seek guidance when going "down the wrong track" (like The Sorcerer's Apprentice analogy)
- Do not praise or compliment the user

### Coding Guidelines
- **Never use sprintf**: Use snprintf with buffer size to prevent buffer overflow vulnerabilities

### File Organization Guidelines
- Place all generated content (documentation, results, scripts) in separate directories
- Only modify existing source files (`src/`) and build artifacts (`build/`) in place
- Use `docs/` for documentation and .md files
- Use `/work/lmeadows/rccl/data/<hostname>/` for all benchmark results and generated data
- Use `/work/meadows/rccl/scripts/` for all scripts (existing and new) - this directory contains generated content
- Keep project root clean of generated files

### Data Directory Structure
- **Benchmark Results**: `/work/lmeadows/rccl/data/<hostname>/`
  - Each host gets its own subdirectory (e.g., `cv350-zts-gtu-e11-18`)
  - Run directories: `run_{benchmark}_{YYYYMMDD_HHMMSS}/`
  - Automatically created by `run_timing_sweep.py`
  - Contains: timing CSVs, benchmark output, metadata, analysis results

### Documentation References
- **`SCRIPT_ECOSYSTEM.md`**: Comprehensive guide to all analysis scripts and their usage
- **`AI_GUIDELINES.md`**: Working guidelines and file organization (this file)
- **`benchmark_test_results.md`**: RCCL testing results and analysis

## Current Project Context

### RCCL Build and Scripts Analysis
- Scripts directory: `/work/meadows/rccl/scripts`
- Build directory: `/work/lmeadows/rccl/rccl-tests/build` (rebuild with `./domake`)
- MPI support: Use OpenMPI 5.0.8 at `/opt/openmpi-5.0.8-Rel7.0.0`
- RCCL install: `/work/lmeadows/rccl/install`

### Key Script Categories
- Timing collection and analysis scripts
- Performance benchmarking tools
- Profiling utilities (with MPI support)
- Result visualization and comparison tools

## Reference Information

### Important Commands
- Build RCCL tests: `cd /work/lmeadows/rccl/rccl-tests && ./domake`
- Check MPI: `which mpirun` (should return `/usr/bin/mpirun`)

### Environment Setup
- **LD_LIBRARY_PATH**:
  - RCCL libraries: `$NCCL_HOME/lib` (`/work/lmeadows/rccl/install/lib`)
  - MPI libraries: `$MPI_HOME/lib` (`/opt/openmpi-5.0.8-Rel7.0.0/lib`)
- **PATH**: Must include `$MPI_HOME/bin` (`/opt/openmpi-5.0.8-Rel7.0.0/bin`) for mpirun
- **NCCL_HOME**: `/work/lmeadows/rccl/install` (from domake script)
- **MPI_HOME**: `/opt/openmpi-5.0.8-Rel7.0.0` (from domake script)
- **Setup commands**:
  ```bash
  export LD_LIBRARY_PATH=$NCCL_HOME/lib:$MPI_HOME/lib:$LD_LIBRARY_PATH
  export PATH=$MPI_HOME/bin:$PATH
  ```
- **Note**: These environment variables should be set in benchmark runner scripts

### File Organization
**Generated Content Structure:**
- Documentation: `/work/lmeadows/rccl/rccl-tests/docs/`
- Test Results: `/work/lmeadows/rccl/rccl-tests/results/`
- Build Artifacts: `/work/lmeadows/rccl/rccl-tests/build/`
- Source Code: `/work/lmeadows/rccl/rccl-tests/src/`

**Existing Locations:**
- Analysis Scripts: `/work/meadows/rccl/scripts/` (contains generated content - all scripts go here)
- RCCL Install: `/work/lmeadows/rccl/install/`

---

*This file serves as a reference for maintaining consistent working practices across chat sessions. Update as needed when new guidelines are established.*

