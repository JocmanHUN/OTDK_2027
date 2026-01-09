# EV Bins Scrolling Feature

## Overview
The EV statistics table in the side panel now displays EV bins in a scrollable format, showing 5 bins at a time instead of all 19 bins.

## Implementation Details

### Data Structure
- **Total bins**: 19 EV ranges (from EV < -50% to EV >= +200%)
- **Visible bins**: 5 at a time in the statistics table
- **Scroll offset**: Tracked in `state.ev_bins_offset` (0 to 14)

### UI Components

#### Statistics Table
- Located in the right-side statistics panel
- Shows EV sáv, Találati arány (Hit rate), Minta (Sample), ROI
- Fixed height of 5 rows
- No vertical scrollbar needed

#### Scroll Buttons
- **◄ Előző**: Scroll to previous 5 bins (decreases offset)
- **Következő ►**: Scroll to next 5 bins (increases offset)
- Located below the EV statistics table
- Always visible when statistics panel is shown

### Key Functions

#### `_scroll_ev_stats(direction: int)`
Adjusts the offset by direction (-1 or +1) and refreshes the statistics display.

#### EV Stats Population
```python
# Get offset and calculate which 5 bins to show
offset = getattr(state, "ev_bins_offset", 0)
start_idx = offset
end_idx = min(offset + 5, len(EV_BINS))

for bin_idx in range(start_idx, end_idx):
    label, _lo, _hi = EV_BINS[bin_idx]
    # ... populate row data
```

### Table Configuration
- **Fixed rows**: Always shows 5 rows (or fewer if at end of list)
- **Column widths**: 
  - EV sáv: 120 pixels
  - Other columns: 90 pixels each
- **Color coding**: 
  - Positive ROI: Dark green (#006400)
  - Negative ROI: Dark red (#8b0000)

## User Experience

### Viewing Statistics
1. Select a prediction model
2. Statistics panel appears on the right
3. EV statistics table shows 5 EV bins
4. Click "◄ Előző" to see previous bins
5. Click "Következő ►" to see next bins
6. Double-click any row to see detailed matches in that EV range

### Navigation
- Offset starts at 0 (showing first 5 bins)
- Maximum offset is 14 (showing last 5 bins: bins 14-18)
- Buttons automatically limit offset to valid range
- Each click shows the next/previous set of 5 bins

## Technical Notes

### State Management
- `AppState.ev_bins_offset: int = 0` - tracks scroll position
- Reset to 0 when changing filters or models
- Persists during statistics refreshes

### Integration Points
- Called from `_update_stats_panel()` in [src/gui/app.py](c:\OTDK_2027\src\gui\app.py#L1406-L1437)
- Updates when `refresh_table()` is called with `allow_network=False`
- Respects `state.exclude_extremes` filter setting

## Benefits
1. **Cleaner interface**: No need for vertical scrollbar
2. **Focused analysis**: Concentrate on specific EV ranges
3. **Easy navigation**: Simple button clicks to explore all bins
4. **Consistent height**: Table size doesn't change, reducing UI jitter
5. **Performance**: Only renders 5 rows at a time

## Difference from Previous Implementation
- Previously: All 19 bins shown in scrollable table
- Now: 5 bins at a time with pagination buttons
- Location: Statistics panel (not main matches table)
- Purpose: Better organization of EV statistics display
