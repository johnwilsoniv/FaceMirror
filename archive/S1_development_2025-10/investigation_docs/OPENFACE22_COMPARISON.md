# OpenFace 2.2 vs 3.0 Comparison Tool

## Purpose
This is a **temporary debugging tool** to identify why S3 Data Analysis works with OpenFace 2.2 but not with OpenFace 3.0.

## What It Does

1. **Finds** all mirrored videos from Face Mirror output
2. **Processes** them through the OpenFace 2.2 binary
3. **Outputs** CSV files in a separate test directory
4. Allows **side-by-side comparison** of OpenFace 2.2 vs 3.0 results

## Usage

### Step 1: Run S1 Face Mirror (OpenFace 3.0)
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror"
python main.py
```
This creates:
- Mirrored videos: `~/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/`
- OpenFace 3.0 CSVs: `~/Documents/SplitFace/S1O Processed Files/Combined Data/`

### Step 2: Run OpenFace 2.2 Comparison
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror"
python run_openface22_comparison.py
```
This creates:
- OpenFace 2.2 CSVs: `~/Documents/SplitFace/S1O Processed Files/OpenFace 2.2 Test/`

### Step 3: Compare Results

**Option A: Direct CSV Comparison**
```bash
# Compare CSV headers
head -1 "~/Documents/SplitFace/S1O Processed Files/Combined Data/video_left_mirrored.csv"
head -1 "~/Documents/SplitFace/S1O Processed Files/OpenFace 2.2 Test/video_left_mirrored.csv"

# Compare AU values
python -c "
import pandas as pd
of3 = pd.read_csv('~/Documents/SplitFace/S1O Processed Files/Combined Data/video_left_mirrored.csv')
of2 = pd.read_csv('~/Documents/SplitFace/S1O Processed Files/OpenFace 2.2 Test/video_left_mirrored.csv')
print('OpenFace 3.0 columns:', of3.columns.tolist())
print('OpenFace 2.2 columns:', of2.columns.tolist())
print('AU columns in OF3:', [c for c in of3.columns if 'AU' in c])
print('AU columns in OF2:', [c for c in of2.columns if 'AU' in c])
"
```

**Option B: Test in S3 Data Analysis**

1. **Backup Combined Data** (OpenFace 3.0 results)
```bash
cp -r "~/Documents/SplitFace/S1O Processed Files/Combined Data" \
      "~/Documents/SplitFace/S1O Processed Files/Combined Data.OF30_backup"
```

2. **Copy OpenFace 2.2 CSVs to Combined Data**
```bash
cp "~/Documents/SplitFace/S1O Processed Files/OpenFace 2.2 Test"/*.csv \
   "~/Documents/SplitFace/S1O Processed Files/Combined Data/"
```

3. **Run S3 Data Analysis with OpenFace 2.2 data**
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis"
python main.py
# Test if it works with OF2.2 data
```

4. **Restore OpenFace 3.0 data**
```bash
cp "~/Documents/SplitFace/S1O Processed Files/Combined Data.OF30_backup"/*.csv \
   "~/Documents/SplitFace/S1O Processed Files/Combined Data/"
```

## Key Directories

| Location | Contains |
|----------|----------|
| `Face Mirror 1.0 Output/` | Mirrored videos (input for both OF2.2 and OF3.0) |
| `Combined Data/` | OpenFace 3.0 CSVs (current pipeline) |
| `OpenFace 2.2 Test/` | OpenFace 2.2 CSVs (comparison output) |

## Expected Outcome

After running this comparison, you should be able to:
1. Identify if CSV formats differ between OF2.2 and OF3.0
2. Compare AU intensity values side-by-side
3. Test if S3 Data Analysis works with OF2.2 data
4. Pinpoint the exact difference causing OF3.0 to fail

## Troubleshooting

**Error: OpenFace 2.2 binary not found**
- Check that the binary exists at: `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction`
- Update `OPENFACE2_BINARY` variable in script if location differs

**Error: No mirrored videos found**
- Verify videos exist in: `~/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/`
- Run S1 Face Mirror first to create mirrored videos

**OpenFace 2.2 processing is very slow**
- This is expected - OpenFace 2.2 is much slower than 3.0
- Consider processing just 1-2 test videos for comparison

## Cleanup

Once debugging is complete, delete this temporary tool:
```bash
rm "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/run_openface22_comparison.py"
rm "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/OPENFACE22_COMPARISON.md"
rm -rf "~/Documents/SplitFace/S1O Processed Files/OpenFace 2.2 Test"
```

## Notes

- **Performance**: OpenFace 2.2 is significantly slower than 3.0 (~20-50x)
- **Temporary**: This is a debugging tool - once the issue is fixed, remove it
- **Binary Requirement**: Requires the OpenFace 2.2 C++ binary to be compiled and available
