#!/bin/bash
# Verification script for GGUF model fix

echo "=================================="
echo "GGUF Model Fix Verification"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}1. Testing GGUF model extraction...${NC}"
/opt/anaconda3/envs/pubmed_py312/bin/python logs/test_gguf_extraction.py

echo ""
echo -e "${GREEN}✓ Verification complete!${NC}"
echo ""
echo "Check the following:"
echo "  1. Model should generate valid JSON (not empty)"
echo "  2. Drugs should be extracted (>0)"
echo "  3. Demographics should include race='Brazilian' and sample_size=102"
echo "  4. No duplicate <|begin_of_text|> warnings"
echo ""
echo "See GGUF_FIX_SUMMARY.md for detailed explanation of the fix."
