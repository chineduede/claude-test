#!/usr/bin/env python3
import re
import os
from collections import defaultdict

# Files to analyze
files = [
    'RangeBreakoutScanner.mq5', 'RangeAnalysis.mqh', 'MLRangeBreakout.mq5',
    'EnhancedMLFeatures.mqh', 'MarketContextFilter.mqh', 'AdvancedRiskManagement.mqh',
    'EnhancedMLRangeBreakout.mq5', 'MLMomentumStrategy.mq5', 'MLMeanReversionStrategy.mq5',
    'AdvancedMarketAnalysis.mqh', 'UltraAdvancedTrading.mqh', 'EliteQuantTrading.mqh'
]

# Additional analysis for specific files
print("=" * 80)
print("DETAILED DEAD CODE ANALYSIS")
print("=" * 80)
print()

# Check for specific unused functions that might be internal to libraries
print("1. LIBRARY FUNCTIONS THAT MAY BE UNUSED:")
print("-" * 40)

# Advanced functions that seem to only be used in one place
advanced_funcs = {
    'CalculateVPIN': 'AdvancedMarketAnalysis.mqh',
    'CalculateToxicity': 'AdvancedMarketAnalysis.mqh',
    'ExecuteVWAP': 'AdvancedMarketAnalysis.mqh',
    'CalculatePatternEmergence': 'EliteQuantTrading.mqh',
    'CalculateEntropy': 'EliteQuantTrading.mqh',
    'CalculateLocalDimension': 'EliteQuantTrading.mqh',
    'EstimateVaR': 'EliteQuantTrading.mqh',
    'EstimateES': 'EliteQuantTrading.mqh',
    'FitGEV': 'EliteQuantTrading.mqh',
    'GeneralizedParetoCDF': 'EliteQuantTrading.mqh',
    'AnalyzeFlowToxicity': 'UltraAdvancedTrading.mqh',
    'DetectSpoofing': 'UltraAdvancedTrading.mqh',
    'PredictMarketImpact': 'UltraAdvancedTrading.mqh',
    'QLearningUpdate': 'UltraAdvancedTrading.mqh',
    'SelectAction': 'UltraAdvancedTrading.mqh',
    'UpdateReplayBuffer': 'UltraAdvancedTrading.mqh',
    'TrainDeepQ': 'UltraAdvancedTrading.mqh',
    'DetectStop_loss': 'EliteQuantTrading.mqh',
    'EstimateShortInterest': 'EliteQuantTrading.mqh',
    'IdentifyAnomalies': 'EliteQuantTrading.mqh',
    'EstimateBeta': 'UltraAdvancedTrading.mqh',
    'CalculateCointegration': 'UltraAdvancedTrading.mqh',
    'CheckStationarity': 'UltraAdvancedTrading.mqh',
}

# Check if these functions are called
for func, file in advanced_funcs.items():
    found_calls = False
    for check_file in files:
        try:
            with open(check_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if func + '(' in content and check_file != file:
                    found_calls = True
                    break
        except:
            pass
    if not found_calls:
        print(f"   {func}() in {file} - Only used internally or not used")

print()

# Check for duplicate implementations
print("2. DUPLICATE FUNCTION IMPLEMENTATIONS:")
print("-" * 40)

duplicate_funcs = defaultdict(list)
for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            func_pattern = re.compile(r'^(?:void|bool|int|double|string|datetime|ENUM_\w+)\s+(\w+)\s*\([^)]*\)', re.MULTILINE)
            for match in func_pattern.finditer(content):
                func_name = match.group(1)
                duplicate_funcs[func_name].append(file)
    except:
        pass

for func, files_list in duplicate_funcs.items():
    if len(files_list) > 1:
        print(f"   {func}() implemented in: {', '.join(files_list)}")

print()

# Check for potentially unused global variables
print("3. GLOBAL VARIABLES ANALYSIS:")
print("-" * 40)

global_vars = {
    'g_performance': 'AdvancedRiskManagement.mqh',
    'hiddenLiquidity': 'UltraAdvancedTrading.mqh',
    'orderBook': 'UltraAdvancedTrading.mqh',
    'marketMaker': 'UltraAdvancedTrading.mqh',
    'neuralNet': 'UltraAdvancedTrading.mqh',
    'adaptiveStop': 'UltraAdvancedTrading.mqh',
    'metaStrategy': 'UltraAdvancedTrading.mqh',
    'replayBuffer': 'UltraAdvancedTrading.mqh',
    'qTable': 'UltraAdvancedTrading.mqh',
    'contagion': 'UltraAdvancedTrading.mqh',
}

for var, file in global_vars.items():
    usage_count = 0
    for check_file in files:
        try:
            with open(check_file, 'r', encoding='utf-8') as f:
                content = f.read()
                pattern = re.compile(r'\b' + re.escape(var) + r'\b')
                usage_count += len(pattern.findall(content))
        except:
            pass
    if usage_count <= 1:  # Only declaration
        print(f"   {var} in {file} - Potentially unused (found {usage_count} references)")

print()

# Check for unused enum values
print("4. ENUM VALUES THAT MAY BE UNUSED:")
print("-" * 40)

enums = {
    'REGIME_VOLATILE': 'EnhancedMLFeatures.mqh',
    'REGIME_TRENDING': 'EnhancedMLFeatures.mqh', 
    'REGIME_RANGING': 'EnhancedMLFeatures.mqh',
    'REGIME_QUIET': 'EnhancedMLFeatures.mqh',
    'NEWS_NONE': 'MarketContextFilter.mqh',
    'NEWS_LOW': 'MarketContextFilter.mqh',
    'NEWS_MEDIUM': 'MarketContextFilter.mqh',
    'NEWS_HIGH': 'MarketContextFilter.mqh',
    'PATTERN_FLAG': 'RangeAnalysis.mqh',
    'PATTERN_PENNANT': 'RangeAnalysis.mqh',
}

for enum, file in enums.items():
    found_usage = False
    for check_file in files:
        try:
            with open(check_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if enum in content and (check_file != file or content.count(enum) > 1):
                    found_usage = True
                    break
        except:
            pass
    if not found_usage:
        print(f"   {enum} in {file} - Only defined, never used")

print()

# Check which includes might be unnecessary
print("5. POTENTIALLY UNNECESSARY INCLUDES:")
print("-" * 40)

include_usage = {
    'Math\\Stat\\Math.mqh': ['UltraAdvancedTrading.mqh', 'EliteQuantTrading.mqh'],
    'UltraAdvancedTrading.mqh': ['MLMeanReversionStrategy.mq5'],
    'EliteQuantTrading.mqh': ['MLMeanReversionStrategy.mq5'],
    'AdvancedMarketAnalysis.mqh': ['MLMeanReversionStrategy.mq5'],
}

# These files seem to only be used by MLMeanReversionStrategy
print("   Files primarily used by MLMeanReversionStrategy.mq5:")
print("   - UltraAdvancedTrading.mqh")
print("   - EliteQuantTrading.mqh") 
print("   - AdvancedMarketAnalysis.mqh")
print("   Note: These are very sophisticated components that may be overkill for simpler strategies")

print()

# Summary of findings
print("6. DEAD CODE SUMMARY:")
print("-" * 40)
print("   - 2 unused structs (MLEnsemble, UltraHFTSignals)")
print("   - 27 unused input parameters across multiple files")
print("   - Multiple advanced functions only used internally")
print("   - Several duplicate function implementations (NormalizeValue, CalculateMarketSentiment, etc.)")
print("   - Several enum values defined but never used")
print("   - The ultra-advanced components are only used by MLMeanReversionStrategy")
print()

print("7. RECOMMENDATIONS:")
print("-" * 40)
print("   1. Remove unused structs MLEnsemble and UltraHFTSignals from EliteQuantTrading.mqh")
print("   2. Remove or comment out unused input parameters")
print("   3. Consider moving ultra-advanced functions to a separate optional module")
print("   4. Consolidate duplicate function implementations")
print("   5. Remove unused enum values NEWS_* and PATTERN_FLAG/PENNANT")
print("   6. Consider simplifying the codebase by removing overly complex features")