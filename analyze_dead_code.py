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

# Storage for analysis
functions = defaultdict(list)  # function_name: [(file, line_no)]
function_calls = defaultdict(list)  # function_name: [(file, line_no)]
structs = defaultdict(list)  # struct_name: [(file, line_no)]
struct_usage = defaultdict(list)  # struct_name: [(file, line_no)]
includes = defaultdict(list)  # included_file: [(file, line_no)]
input_params = defaultdict(list)  # param_name: [(file, line_no)]
input_usage = defaultdict(list)  # param_name: [(file, line_no)]
global_vars = defaultdict(list)  # var_name: [(file, line_no, type)]
global_var_usage = defaultdict(list)  # var_name: [(file, line_no)]

# Patterns
func_pattern = re.compile(r'^(?:void|bool|int|double|string|datetime|ENUM_\w+)\s+(\w+)\s*\([^)]*\)', re.MULTILINE)
func_call_pattern = re.compile(r'(\w+)\s*\(')
struct_pattern = re.compile(r'^struct\s+(\w+)', re.MULTILINE)
struct_usage_pattern = re.compile(r'(\w+)\s+\w+[;\[]|(\w+)\s*&\s*\w+')
include_pattern = re.compile(r'#include\s+[<"]([^>"]+)[>"]')
input_pattern = re.compile(r'^input\s+\w+\s+(\w+)', re.MULTILINE)
global_var_pattern = re.compile(r'^(?!input\s)(?:static\s+)?(\w+)\s+(\w+)(?:\[\])?(?:\s*=|;)', re.MULTILINE)

# Read and analyze each file
for filename in files:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Find function definitions
            for match in func_pattern.finditer(content):
                func_name = match.group(1)
                line_no = content[:match.start()].count('\n') + 1
                functions[func_name].append((filename, line_no))
            
            # Find function calls
            for i, line in enumerate(lines):
                # Skip comments and strings
                if '//' in line:
                    line = line[:line.index('//')]
                if '/*' in line:
                    continue
                    
                for match in func_call_pattern.finditer(line):
                    func_name = match.group(1)
                    # Skip keywords and types
                    if func_name not in ['if', 'for', 'while', 'switch', 'return', 'new', 
                                       'void', 'bool', 'int', 'double', 'string', 'datetime',
                                       'ArrayResize', 'ArraySize', 'Print', 'MathAbs', 'MathMin',
                                       'MathMax', 'MathRand', 'MathExp', 'MathRound', 'MathPow',
                                       'StringFind', 'TimeToStruct', 'TimeCurrent', 'ObjectCreate',
                                       'ObjectSetInteger', 'ObjectSetString', 'ObjectsDeleteAll',
                                       'FileOpen', 'FileClose', 'FileWrite', 'FileRead',
                                       'FileWriteStruct', 'FileReadStruct', 'FileIsExist',
                                       'SymbolInfoDouble', 'SymbolInfoInteger', 'SymbolSelect',
                                       'iTime', 'iOpen', 'iHigh', 'iLow', 'iClose', 'iVolume',
                                       'iBarShift', 'iATR', 'iMA', 'iRSI', 'iMomentum',
                                       'CopyBuffer', 'CopyTickVolume', 'DoubleToString',
                                       'IntegerToString', 'TimeToString', 'DBL_MAX', 'INVALID_HANDLE']:
                        function_calls[func_name].append((filename, i + 1))
            
            # Find struct definitions
            for match in struct_pattern.finditer(content):
                struct_name = match.group(1)
                line_no = content[:match.start()].count('\n') + 1
                structs[struct_name].append((filename, line_no))
            
            # Find struct usage
            for i, line in enumerate(lines):
                if '//' in line:
                    line = line[:line.index('//')]
                for match in struct_usage_pattern.finditer(line):
                    struct_name = match.group(1) or match.group(2)
                    if struct_name in structs:
                        struct_usage[struct_name].append((filename, i + 1))
            
            # Find includes
            for match in include_pattern.finditer(content):
                included = match.group(1)
                line_no = content[:match.start()].count('\n') + 1
                includes[included].append((filename, line_no))
            
            # Find input parameters
            for match in input_pattern.finditer(content):
                param_name = match.group(1)
                line_no = content[:match.start()].count('\n') + 1
                input_params[param_name].append((filename, line_no))
            
            # Find input parameter usage
            for param in input_params:
                pattern = re.compile(r'\b' + re.escape(param) + r'\b')
                for i, line in enumerate(lines):
                    if pattern.search(line) and not line.strip().startswith('input'):
                        input_usage[param].append((filename, i + 1))
            
            # Find global variables
            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith('//') and not line.startswith('input'):
                    match = global_var_pattern.match(line)
                    if match and not any(line.startswith(x) for x in ['struct', 'enum', 'class', '#']):
                        var_type = match.group(1)
                        var_name = match.group(2)
                        if var_type not in ['void', 'struct', 'enum', 'class'] and var_name not in functions:
                            global_vars[var_name].append((filename, i + 1, var_type))
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Analyze unused code
print("=" * 80)
print("DEAD CODE ANALYSIS REPORT")
print("=" * 80)
print()

# 1. Unused Functions
print("1. UNUSED FUNCTIONS:")
print("-" * 40)
unused_funcs = []
for func, defs in functions.items():
    if func not in function_calls or not function_calls[func]:
        for file, line in defs:
            unused_funcs.append((func, file, line))

if unused_funcs:
    for func, file, line in sorted(unused_funcs):
        print(f"   {func}() - {file}:{line}")
else:
    print("   No unused functions found")
print()

# 2. Unused Structs
print("2. UNUSED STRUCTS:")
print("-" * 40)
unused_structs = []
for struct, defs in structs.items():
    if struct not in struct_usage or not struct_usage[struct]:
        for file, line in defs:
            unused_structs.append((struct, file, line))

if unused_structs:
    for struct, file, line in sorted(unused_structs):
        print(f"   {struct} - {file}:{line}")
else:
    print("   No unused structs found")
print()

# 3. Unused Input Parameters
print("3. UNUSED INPUT PARAMETERS:")
print("-" * 40)
unused_inputs = []
for param, defs in input_params.items():
    if param not in input_usage or len(input_usage[param]) <= 1:  # Only definition
        for file, line in defs:
            unused_inputs.append((param, file, line))

if unused_inputs:
    for param, file, line in sorted(unused_inputs):
        print(f"   {param} - {file}:{line}")
else:
    print("   No unused input parameters found")
print()

# 4. Include Analysis
print("4. INCLUDE FILES ANALYSIS:")
print("-" * 40)
for inc, usages in sorted(includes.items()):
    print(f"   {inc}:")
    for file, line in usages:
        print(f"      Used in: {file}:{line}")
print()

# 5. Cross-file function usage
print("5. CROSS-FILE FUNCTION USAGE:")
print("-" * 40)
for func, calls in function_calls.items():
    if func in functions:
        def_files = set(f[0] for f in functions[func])
        call_files = set(f[0] for f in calls)
        if def_files and call_files and def_files != call_files:
            print(f"   {func}():")
            print(f"      Defined in: {', '.join(def_files)}")
            print(f"      Called from: {', '.join(call_files)}")
print()

# 6. Summary
print("SUMMARY:")
print("-" * 40)
print(f"Total functions defined: {sum(len(defs) for defs in functions.values())}")
print(f"Unused functions: {len(unused_funcs)}")
print(f"Total structs defined: {sum(len(defs) for defs in structs.values())}")
print(f"Unused structs: {len(unused_structs)}")
print(f"Total input parameters: {sum(len(defs) for defs in input_params.values())}")
print(f"Unused input parameters: {len(unused_inputs)}")