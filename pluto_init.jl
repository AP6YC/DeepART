"""
    pluto_init.jl

# Description
Convenience script for loading and running Pluto for the project wrapped in a Revise call.
"""

# Run Revise first
using Revise

# Load Pluto as a dependency
using Pluto

# Initialize the Pluto kernel
Pluto.run()
