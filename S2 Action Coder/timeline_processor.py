# timeline_processor.py

import string
# Removed unused import: import config
from thefuzz import fuzz # Import fuzz
import math # Make sure math is imported if needed for calculations (e.g. floor/ceil, though not currently used)
# Import config locally where needed or pass relevant values during init
import config as ConfigModule # Import the module itself
import copy # Import copy

class TimelineProcessor:
    """
    Handles processing of WhisperX segments to identify commands, near misses,
    resolve overlaps, and generate action ranges. Includes O/E sequence override
    and handles SO_SE code from config mappings with stricter matching.
    Applies stricter matching to standalone "oh" command.
    Generates placeholder ranges for Near Misses ("NM").
    Adds status field for confirmation needed ranges.
    """

    def __init__(self, config_module):
        self.config = config_module # Store config module passed during init
        self.FUZZY_SEQUENCE_THRESHOLD = getattr(config_module, 'FUZZY_SEQUENCE_THRESHOLD', 91)
        self.NEAR_MISS_THRESHOLD = getattr(config_module, 'NEAR_MISS_THRESHOLD', 70)
        self.HIGH_CONFIDENCE_THRESHOLD = getattr(config_module, 'HIGH_CONFIDENCE_THRESHOLD', 95)
        self.STRICT_MATCH_THRESHOLD = getattr(config_module, 'STRICT_MATCH_THRESHOLD', 95) # For SO_SE config match & standalone 'oh'
        self.MERGE_TIME_THRESHOLD_SECONDS = getattr(config_module, 'MERGE_TIME_THRESHOLD_SECONDS', 0.25)
        self.MERGE_PAIRS = getattr(config_module, 'MERGE_PAIRS', {("SS", "BS"): "BS"})
        self._compile_command_phrase_map()
        self.O_WORDS = {'o', 'oh', 'o.'}
        self.E_WORDS = {'e', 'eee', 'ee', 'e.'}
        self.MAX_OE_WORD_GAP = 2
        self.MAX_OE_TIME_GAP_S = 1.0
        self.OE_OVERRIDE_SCORE = 102
        self.OE_CONFIG_SCORE = 101.5


    def _compile_command_phrase_map(self):
        # (No changes needed)
        self.command_phrase_map = {}
        voice_mappings = getattr(self.config, 'VOICE_COMMAND_MAPPINGS', {})
        for phrase, code in voice_mappings.items():
            normalized_phrase = phrase.lower().translate(str.maketrans('', '', string.punctuation))
            phrase_words = normalized_phrase.split()
            if phrase_words:
                self.command_phrase_map[phrase] = {'code': code, 'words': phrase_words}
        self.sorted_command_phrases_for_split = sorted(
            self.command_phrase_map.items(),
            key=lambda item: len(item[1]['words']),
            reverse=True
        )


    def process_segments(self, whisper_segments, fps, total_frames):
        # (No changes needed)
        if fps <= 0 or total_frames <= 0:
            print("TimelineProcessor ERROR: Invalid video properties (FPS/Frames). Aborting.")
            return [], []
        print(f"TimelineProcessor: Starting processing for {len(whisper_segments)} segments.")
        print(f"TimelineProcessor: Step 1: Event Identification (FPS: {fps:.3f})...")
        all_potential_events = []
        failed_events = 0
        for i, segment in enumerate(whisper_segments):
            if 'words' not in segment or not isinstance(segment['words'], list):
                print(f"TimelineProcessor WARN: Segment {i} missing or invalid 'words' list. Skipping.")
                continue
            events_in_segment = self._split_segment_into_commands_hybrid_oe(segment)
            for ev in events_in_segment:
                start_time, end_time = ev.get('start_time'), ev.get('end_time')
                # --- Convert numpy floats to standard floats for safety ---
                if start_time is not None: start_time = float(start_time)
                if end_time is not None: end_time = float(end_time)
                ev['start_time'] = start_time # Update dict
                ev['end_time'] = end_time   # Update dict
                # --- End Conversion ---
                if start_time is not None and end_time is not None and isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)) and start_time >= 0 and end_time >= start_time:
                    ev['start_frame'] = int(start_time * fps); ev['end_frame'] = max(ev['start_frame'], int(end_time * fps))
                else:
                    ev['start_frame'], ev['end_frame'] = None, None; failed_events += 1
                    print(f"TimelineProcessor WARN: Event missing/invalid time data: {ev.get('phrase', ev.get('transcribed_text'))}")
            if events_in_segment: all_potential_events.extend(events_in_segment)
        print(f"TimelineProcessor: Step 1: Found {len(all_potential_events)} potential events initially. ({failed_events} skipped due to time data).")
        print("TimelineProcessor: Step 2: Merging consecutive commands...")
        merged_timeline_events = self._merge_consecutive_commands(all_potential_events)
        print("TimelineProcessor: Step 3: Finalizing timeline (overlap resolution)...")
        final_timeline_events = self._finalize_timeline(merged_timeline_events)
        num_cmds = sum(1 for e in final_timeline_events if e['type'] == 'command')
        num_nms = sum(1 for e in final_timeline_events if e['type'] == 'near_miss')
        print(f"TimelineProcessor: Step 3: Final Timeline has {num_cmds} commands, {num_nms} near misses.")
        print("TimelineProcessor: Step 4: Generating action ranges...")
        generated_action_ranges = self._generate_action_ranges(final_timeline_events, fps, total_frames)
        print("TimelineProcessor: Processing complete.")
        return final_timeline_events, generated_action_ranges


    def _split_segment_into_commands_hybrid_oe(self, segment_data):
         # (No changes needed)
        segment_id = segment_data.get('id', 'Unknown')
        if not segment_data or 'words' not in segment_data or not segment_data['words']: return []
        segment_words_data = segment_data['words']
        all_potential_events = []
        processed_word_indices = set()

        LT_KEYWORDS = ["bottom", "lower"]; ET_KEYWORDS = ["tight", "firmly"]; AMBIGUOUS_BS_PHRASES = ["show me your teeth"]

        o_found_at = -1; o_word_data = None
        for idx, word_info in enumerate(segment_words_data):
            if idx in processed_word_indices: o_found_at = -1; continue
            word_lower = word_info.get('word', '').lower()
            if word_lower in self.O_WORDS: o_found_at = idx; o_word_data = word_info; continue
            if o_found_at != -1 and word_lower in self.E_WORDS:
                e_word_data = word_info; word_gap = idx - o_found_at - 1; time_gap = float('inf')
                o_end, e_start = o_word_data.get('end'), e_word_data.get('start')
                if o_end is not None: o_end = float(o_end)
                if e_start is not None: e_start = float(e_start)
                if o_end is not None and e_start is not None: time_gap = e_start - o_end
                if 0 <= word_gap <= self.MAX_OE_WORD_GAP and 0 <= time_gap <= self.MAX_OE_TIME_GAP_S:
                    start_time = float(o_word_data.get('start', 0.0))
                    end_time = float(e_word_data.get('end', start_time))
                    if start_time is not None and end_time is not None:
                        sequence_indices = set(range(o_found_at, idx + 1))
                        sequence_words = segment_words_data[o_found_at : idx + 1]
                        event_data = { 'type': 'command', 'code': 'SO_SE', 'phrase': f"O->E Sequence Override ({o_word_data.get('word')}...{e_word_data.get('word')})", 'start_time': float(start_time), 'end_time': float(end_time), 'score': self.OE_OVERRIDE_SCORE, 'start_index': o_found_at, 'length': idx - o_found_at + 1, 'original_text': " ".join([w['word'] for w in sequence_words]), 'matched_words': sequence_words }
                        all_potential_events.append(event_data)
                        processed_word_indices.update(sequence_indices)
                    o_found_at = -1; o_word_data = None; continue
                else:
                    o_found_at = -1; o_word_data = None
                    if word_lower in self.O_WORDS: o_found_at = idx; o_word_data = word_info
            elif o_found_at != -1 and word_lower not in self.E_WORDS:
                o_found_at = -1; o_word_data = None
            if o_found_at == -1 and word_lower in self.O_WORDS:
                 o_found_at = idx; o_word_data = word_info

        for phrase_key, command_info in self.sorted_command_phrases_for_split:
            command_words_target = command_info['words']; code = command_info['code']; len_command = len(command_words_target)
            if len_command == 0: continue
            normalized_target_phrase_str = " ".join(command_words_target).lower().translate(str.maketrans('', '', string.punctuation))

            for i in range(len(segment_words_data) - len_command + 1):
                segment_sublist_words_data = segment_words_data[i : i + len_command]
                current_indices = set(range(i, i + len_command))
                if not current_indices.isdisjoint(processed_word_indices): continue
                if not all(isinstance(w.get('word'), str) and w.get('start') is not None and w.get('end') is not None for w in segment_sublist_words_data): continue
                start_time = float(segment_sublist_words_data[0]['start'])
                end_time = float(segment_sublist_words_data[-1]['end'])
                if not (isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)) and start_time >= 0 and end_time >= start_time): continue

                segment_words_list_raw = [w['word'] for w in segment_sublist_words_data]
                segment_phrase_str_raw = " ".join(segment_words_list_raw)
                segment_words_list_norm = [w.lower().translate(str.maketrans('', '', string.punctuation)) for w in segment_words_list_raw]
                segment_phrase_str_norm = " ".join(segment_words_list_norm)

                is_exact_match = (segment_phrase_str_norm == normalized_target_phrase_str)
                score = 0

                if code == 'SO_SE':
                    strict_score = fuzz.ratio(normalized_target_phrase_str, segment_phrase_str_norm)
                    if strict_score >= self.STRICT_MATCH_THRESHOLD: score = strict_score
                elif code == 'SO' and phrase_key == 'oh':
                    if len(segment_words_list_norm) == 1:
                        ratio_score = fuzz.ratio(normalized_target_phrase_str, segment_phrase_str_norm)
                        if ratio_score >= self.STRICT_MATCH_THRESHOLD: score = ratio_score
                else:
                    score = 101 if is_exact_match else fuzz.token_set_ratio(normalized_target_phrase_str, segment_phrase_str_raw)

                event_data = None
                threshold_to_use = self.STRICT_MATCH_THRESHOLD if (code == 'SO' and phrase_key == 'oh') else self.FUZZY_SEQUENCE_THRESHOLD

                if score > 0 and score >= threshold_to_use:
                    corrected_code = code; log_phrase = phrase_key; event_score = score
                    if code == 'SO_SE': event_score = self.OE_CONFIG_SCORE
                    elif code == 'SO' and phrase_key == 'oh': print(f"  Standalone 'oh' Check: PASSED (Score {score}>={threshold_to_use}). Phrase='{phrase_key}', Text='{segment_phrase_str_norm}'")
                    if not (code == 'SO' and phrase_key == 'oh'):
                        if code == 'BS' and phrase_key in AMBIGUOUS_BS_PHRASES:
                            if any(kw in segment_phrase_str_raw.lower() for kw in LT_KEYWORDS): corrected_code = 'LT'; log_phrase += " (->LT)"
                        elif code == 'ES':
                            if any(kw in segment_phrase_str_raw.lower() for kw in ET_KEYWORDS): corrected_code = 'ET'; log_phrase += " (->ET)"; event_score = self.OE_OVERRIDE_SCORE
                    event_data = { 'type': 'command', 'code': corrected_code, 'phrase': log_phrase, 'start_time': float(start_time), 'end_time': float(end_time), 'score': event_score, 'start_index': i, 'length': len_command, 'original_text': segment_phrase_str_raw, 'matched_words': segment_sublist_words_data }
                elif not (code == 'SO' and phrase_key == 'oh') and score >= self.NEAR_MISS_THRESHOLD:
                     event_data = { 'type': 'near_miss', 'transcribed_text': segment_phrase_str_raw, 'start_time': float(start_time), 'end_time': float(end_time), 'score': score, 'start_index': i, 'length': len_command, 'matched_words': segment_sublist_words_data }

                if event_data: all_potential_events.append(event_data)

        if not all_potential_events: return []
        def get_priority(event):
            score = event.get('score', 0)
            if event['type'] == 'command':
                if score == self.OE_OVERRIDE_SCORE: return 0
                elif score == self.OE_CONFIG_SCORE: return 1
                elif score >= 101: return 2
                else: return 3
            elif event['type'] == 'near_miss': return 4
            else: return 5
        sorted_potential = sorted(all_potential_events, key=lambda event: ( get_priority(event), -event.get('score', 0), -event.get('length', 0), event['start_index'] ))
        final_segment_events = []; final_processed_indices = set()
        for event in sorted_potential:
            event_indices = set(range(event['start_index'], event['start_index'] + event['length']))
            if not event_indices.isdisjoint(final_processed_indices): continue
            final_segment_events.append(event); final_processed_indices.update(event_indices)
        final_segment_events.sort(key=lambda x: x['start_time'])
        return final_segment_events

    def _merge_consecutive_commands(self, potential_events):
        # (No changes needed)
        if not potential_events: return []
        sorted_events = sorted([ev for ev in potential_events if ev.get('start_frame') is not None], key=lambda x: x['start_frame'])
        merged_list = []; i = 0
        while i < len(sorted_events):
            current_event = sorted_events[i]
            if i + 1 < len(sorted_events) and current_event['type'] == 'command':
                next_event = sorted_events[i+1]
                if next_event['type'] == 'command':
                    time_gap = next_event['start_time'] - current_event['end_time']
                    code_pair = (current_event['code'], next_event['code'])
                    if code_pair in self.MERGE_PAIRS and 0 <= time_gap < self.MERGE_TIME_THRESHOLD_SECONDS:
                        res_code = self.MERGE_PAIRS[code_pair]; phrase = f"{current_event.get('phrase','?')}({current_event['code']}) >> {next_event.get('phrase','?')}({next_event['code']})"
                        score = next_event.get('score', 0); trigger_start = current_event.get('trigger_start', current_event.get('start_time')); trigger_end = current_event.get('trigger_end', current_event.get('end_time')); trigger_start_frame = current_event.get('trigger_start_frame', current_event.get('start_frame')); trigger_end_frame = current_event.get('trigger_end_frame', current_event.get('end_frame')); matched_words = current_event.get('matched_words')
                        merged = { 'type': 'command', 'code': res_code, 'phrase': phrase, 'start_time': current_event['start_time'], 'end_time': next_event['end_time'], 'start_frame': current_event['start_frame'],'end_frame': next_event['end_frame'], 'score': score, 'trigger_start': trigger_start, 'trigger_end': trigger_end, 'trigger_start_frame': trigger_start_frame, 'trigger_end_frame': trigger_end_frame, 'start_index': None, 'length': None, 'matched_words': matched_words }
                        merged_list.append(merged); i += 2; continue
                    elif current_event['code'] == next_event['code'] and 0 <= time_gap < self.MERGE_TIME_THRESHOLD_SECONDS:
                        res_code = current_event['code']; phrase = f"{current_event.get('phrase','?')}({res_code}) + ..."
                        score = max(current_event.get('score', 0), next_event.get('score', 0)); trigger_start = current_event.get('trigger_start', current_event.get('start_time')); trigger_end = current_event.get('trigger_end', current_event.get('end_time')); trigger_start_frame = current_event.get('trigger_start_frame', current_event.get('start_frame')); trigger_end_frame = current_event.get('trigger_end_frame', current_event.get('end_frame')); matched_words = current_event.get('matched_words')
                        merged = { 'type': 'command', 'code': res_code, 'phrase': phrase, 'start_time': current_event['start_time'], 'end_time': next_event['end_time'], 'start_frame': current_event['start_frame'],'end_frame': next_event['end_frame'], 'score': score, 'trigger_start': trigger_start, 'trigger_end': trigger_end, 'trigger_start_frame': trigger_start_frame, 'trigger_end_frame': trigger_end_frame, 'start_index': None, 'length': None, 'matched_words': matched_words }
                        merged_list.append(merged); i += 2; continue
            merged_list.append(current_event); i += 1
        return merged_list


    # --- MODIFIED: _finalize_timeline with improved overlap check ---
    def _finalize_timeline(self, merged_events):
        if not merged_events:
            return []

        def get_priority(event):
            """Assigns a numerical priority (lower is better)."""
            score = event.get('score', 0)
            event_type = event.get('type')
            if event_type == 'command':
                if score == self.OE_OVERRIDE_SCORE: return 0  # Highest priority: O->E seq override
                elif score == self.OE_CONFIG_SCORE: return 1  # High priority: SO_SE from config
                elif score >= 101: return 2                  # Exact match
                elif score >= self.HIGH_CONFIDENCE_THRESHOLD: return 3 # High confidence fuzzy match
                else: return 4                              # Lower confidence fuzzy (confirm needed)
            elif event_type == 'near_miss': return 5          # Near Miss
            else: return 6                                  # Unknown / Fallback

        # Sort events by priority (lower number first), then by start frame,
        # then by score (higher first), then by length (longer first) as tie-breakers
        sorted_events = sorted(merged_events, key=lambda x: (
            get_priority(x),
            x.get('start_frame', 0),
            -x.get('score', 0),
            -(x.get('end_frame', x.get('start_frame', 0)) - x.get('start_frame', 0))
        ))

        final_timeline = []
        # Store tuples: (start_frame, end_frame, event_object)
        processed_intervals = []

        for event in sorted_events:
            event_start = event.get('start_frame'); event_end = event.get('end_frame')
            if event_start is None or event_end is None:
                print(f"TimelineProcessor WARN (Finalize): Skipping event due to missing frame data: {event.get('phrase', event.get('transcribed_text'))}"); continue

            should_skip = False
            event_priority = get_priority(event)

            for proc_start, proc_end, proc_event_ref in processed_intervals:
                # Basic check: If the current event is entirely contained within or identical to a processed interval
                # if proc_start <= event_start and event_end <= proc_end:
                #    should_skip = True # Simple containment check first
                #    break

                # More precise overlap check: (StartA <= EndB) and (EndA >= StartB)
                if not (event_end < proc_start or event_start > proc_end):
                    # --- Overlap detected, now check priority ---
                    processed_priority = get_priority(proc_event_ref)

                    # Skip the current event *only if* its priority is strictly worse (higher number)
                    # than the priority of the event already processed in this overlapping interval.
                    if event_priority > processed_priority:
                        should_skip = True
                        overlapping_event_details = (f"Overlap with: Type='{proc_event_ref.get('type')}' "
                                                     f"Phrase='{proc_event_ref.get('phrase', proc_event_ref.get('transcribed_text', '??'))}' "
                                                     f"(F{proc_start}-{proc_end}, Prio {processed_priority}, Score {proc_event_ref.get('score', 0):.1f})")
                        print(f"TimelineProcessor INFO (Finalize): Skipping Event: Type='{event.get('type')}' "
                              f"Phrase='{event.get('phrase', event.get('transcribed_text', '??'))}' "
                              f"(F{event_start}-{event_end}, Prio {event_priority}, Score {event.get('score', 0):.1f}). {overlapping_event_details}")
                        break # Found a higher-priority overlap, no need to check further
                    # else: # Optional: Log when overlap is detected but not skipped due to priority
                    #    print(f"TimelineProcessor DEBUG (Finalize): Overlap detected but NOT skipping (Prio {event_priority} <= {processed_priority}): Event='{event.get('phrase', event.get('transcribed_text', '??'))}' (F{event_start}-{event_end}) vs Processed='{proc_event_ref.get('phrase', proc_event_ref.get('transcribed_text', '??'))}' (F{proc_start}-{proc_end})")

            if not should_skip:
                final_timeline.append(event)
                # Store the event itself along with interval
                processed_intervals.append((event_start, event_end, event))
                # Keep processed_intervals sorted by start frame for potentially faster overlap checks later (optional optimization)
                processed_intervals.sort(key=lambda x: x[0])

        # Final sort of the accepted timeline events by start frame
        final_timeline.sort(key=lambda x: x.get('start_frame', 0));
        return final_timeline


    # --- _generate_action_ranges (No changes needed here) ---
    def _generate_action_ranges(self, final_timeline_events, fps, total_video_frames):
        generated_action_ranges = []
        relevant_events = sorted(
            [event for event in final_timeline_events if event.get('type') in ['command', 'near_miss'] and event.get('start_frame') is not None],
            key=lambda x: x['start_frame']
        )
        events_for_ranges = []
        for i, event in enumerate(relevant_events):
            if event['type'] == 'command' and event.get('code') == 'STOP': continue
            events_for_ranges.append(event)

        if not events_for_ranges:
            print("  No non-STOP commands or near misses found. Generating full baseline.")
            baseline_range = { 'action_code': 'BL', 'start_time': 0.0, 'end_time': (total_video_frames - 1) / fps if fps > 0 else 0.0, 'start_frame': 0, 'end_frame': total_video_frames - 1, 'trigger_phrase': 'No Actions Found', 'trigger_start': 0.0, 'trigger_end': 0.0, 'trigger_start_frame': 0, 'trigger_end_frame': 0, 'confidence_score': 101, 'index': 0, 'status': None, 'matched_words': None }
            return [baseline_range]

        range_index_counter = 0; last_range_end_frame = -1
        first_event_start_frame = events_for_ranges[0].get('start_frame')
        if first_event_start_frame > 0:
            initial_baseline_end_frame = first_event_start_frame - 1
            bl_end_time = initial_baseline_end_frame / fps if fps > 0 else 0.0
            baseline = { 'action_code': 'BL', 'start_time': 0.0, 'end_time': bl_end_time, 'start_frame': 0, 'end_frame': initial_baseline_end_frame, 'trigger_phrase': 'Implied Baseline Start', 'trigger_start': 0.0, 'trigger_end': 0.0, 'trigger_start_frame': 0, 'trigger_end_frame': 0, 'confidence_score': 101, 'index': range_index_counter, 'status': None, 'matched_words': None }
            generated_action_ranges.append(baseline);
            range_index_counter += 1; last_range_end_frame = initial_baseline_end_frame

        for i, current_event in enumerate(events_for_ranges):
            event_type = current_event.get('type'); event_start_frame = current_event.get('start_frame'); event_min_end_frame = current_event.get('end_frame', event_start_frame)
            original_event_start_time = current_event.get('start_time')
            original_event_end_time = current_event.get('end_time')

            code = 'UNKNOWN'; status = None; confidence_score = current_event.get('score', 0)
            trigger_phrase = current_event.get('phrase', current_event.get('transcribed_text', '?'))
            if event_type == 'command':
                code = current_event.get('code', 'UNKNOWN')
                if self.FUZZY_SEQUENCE_THRESHOLD <= confidence_score < self.HIGH_CONFIDENCE_THRESHOLD: status = 'confirm_needed'; print(f"  Event {i}: Command '{code}' marked for confirmation (Score: {confidence_score:.1f})")
            elif event_type == 'near_miss': code = "NM"; status = None; print(f"  Event {i}: Near Miss detected (Score: {confidence_score:.1f})")

            trigger_start = current_event.get('trigger_start', original_event_start_time);
            trigger_end = current_event.get('trigger_end', original_event_end_time);
            trigger_start_frame_orig = current_event.get('trigger_start_frame', event_start_frame);
            trigger_end_frame_orig = current_event.get('trigger_end_frame', event_min_end_frame);
            matched_words = current_event.get('matched_words')

            current_start_frame_for_range = max(event_start_frame, last_range_end_frame + 1)

            # NOTE: Gaps between actions are left empty (no BL insertion)
            # BL only represents truly neutral baseline at start, not gaps from STOP commands

            if current_start_frame_for_range > event_min_end_frame: print(f"  Range SKIPPED (Calc StartF {current_start_frame_for_range} > Event MinEndF {event_min_end_frame}): Code='{code}'"); continue

            perform_oe_split = False; so_end_frame = -1; se_start_frame = -1; T_o_end = None; T_e_start = None; split_source = None
            if code == 'SO_SE':
                print(f"  O/E Split Check: SO_SE code found for event at F{event_start_frame}"); perform_oe_split = True; split_source = 'code/override'
                if matched_words and len(matched_words) > 1:
                     o_indices = [idx for idx, w in enumerate(matched_words) if w.get('word', '').lower() in self.O_WORDS]; e_indices = [idx for idx, w in enumerate(matched_words) if w.get('word', '').lower() in self.E_WORDS]
                     if o_indices and e_indices and max(o_indices) < min(e_indices): T_o_end = matched_words[max(o_indices)].get('end'); T_e_start = matched_words[min(e_indices)].get('start'); print(f"    Split (SO_SE): Using word timings. T_o_end={T_o_end:.3f}, T_e_start={T_e_start:.3f}")
                     else: T_o_end, T_e_start = None, None; print(f"    Split (SO_SE): Word pattern O->E not found/clear in matched words.")
                else: T_o_end, T_e_start = None, None; print(f"    Split (SO_SE): Matched words missing or insufficient.")
                if T_o_end is not None and T_e_start is not None and fps > 0: so_end_frame = int(float(T_o_end) * fps); se_start_frame = int(float(T_e_start) * fps)
                else:
                     print("    Split (SO_SE): Falling back to heuristic duration split based on trigger frames.")
                     split_base_start = trigger_start_frame_orig if trigger_start_frame_orig is not None else event_start_frame; split_base_end = trigger_end_frame_orig if trigger_end_frame_orig is not None else event_min_end_frame
                     mid_frame = split_base_start + (split_base_end - split_base_start) // 2; so_end_frame = mid_frame; se_start_frame = mid_frame + 1
                     if fps > 0: T_o_end = so_end_frame / fps; T_e_start = se_start_frame / fps
                if se_start_frame <= so_end_frame: print(f"    O/E Split WARN: Adjusting SE start F{se_start_frame} to {so_end_frame + 1} (SO End was {so_end_frame})."); se_start_frame = so_end_frame + 1;
                if fps > 0: T_e_start = se_start_frame / fps
                print(f"    O/E Split Final Frames (Source: {split_source}): SO End F{so_end_frame}, SE Start F{se_start_frame}")

            next_event_start_frame = total_video_frames
            next_event_details = "End of Video"
            for k in range(relevant_events.index(current_event) + 1, len(relevant_events)):
                 next_event = relevant_events[k]; next_start = next_event.get('start_frame')
                 if next_start is not None:
                     if next_event.get('type') == 'command' and next_event.get('code') == 'STOP':
                          next_event_start_frame = next_start
                          next_event_details = f"Next Event: Type={next_event.get('type')} Code={next_event.get('code')} StartF={next_start}"
                          break
                     elif next_event.get('type') in ['command', 'near_miss']:
                          next_event_start_frame = next_start
                          next_event_details = f"Next Event: Type={next_event.get('type')} Code={next_event.get('code', 'NM')} StartF={next_start}"
                          break

            potential_end_frame = next_event_start_frame - 1
            range_end_frame = max(event_min_end_frame, potential_end_frame)
            range_end_frame_final = min(max(current_start_frame_for_range, range_end_frame), total_video_frames - 1)

            # --- Log Range Calculation Steps ---
            print(f"  Range Calc for Event {i} ('{code}', F{event_start_frame}-{event_min_end_frame}):")
            print(f"    - Start Frame Used: {current_start_frame_for_range} (EventStart={event_start_frame}, LastEnd={last_range_end_frame})")
            print(f"    - Min End Frame (from event words): {event_min_end_frame}")
            print(f"    - Next Relevant Event Start: F{next_event_start_frame} ({next_event_details})")
            print(f"    - Potential End Frame (NextStart-1): {potential_end_frame}")
            print(f"    - Max(MinEnd, PotentialEnd): {range_end_frame}")
            print(f"    - Final Range Frames: F{current_start_frame_for_range} - F{range_end_frame_final}")
            # --- End Log ---

            if perform_oe_split and so_end_frame >= current_start_frame_for_range and se_start_frame > so_end_frame:
                so_end_clamp = min(so_end_frame, range_end_frame_final); so_end_time = so_end_clamp / fps if fps > 0 else 0.0
                if so_end_clamp >= current_start_frame_for_range:
                    so_range = { 'action_code': 'SO', 'start_frame': current_start_frame_for_range, 'end_frame': so_end_clamp,
                                 'start_time': current_start_frame_for_range / fps if fps > 0 else 0.0, 'end_time': so_end_time,
                                 'trigger_phrase': trigger_phrase + " (Split O)",
                                 'trigger_start': trigger_start, 'trigger_end': T_o_end if T_o_end is not None else so_end_time,
                                 'trigger_start_frame': trigger_start_frame_orig, 'trigger_end_frame': so_end_frame,
                                 'confidence_score': confidence_score, 'index': range_index_counter, 'status': status, 'matched_words': matched_words,
                                 'original_event_start_time': original_event_start_time,
                                 'original_event_end_time': original_event_end_time
                                 }
                    generated_action_ranges.append(so_range);
                    last_range_end_frame = so_end_clamp; range_index_counter += 1
                else: print(f"  Range SKIPPED (Split SO EndF {so_end_clamp} < StartF {current_start_frame_for_range})")

                se_start_clamp = max(se_start_frame, last_range_end_frame + 1); se_end_final = range_end_frame_final; se_end_time = se_end_final / fps if fps > 0 else 0.0
                if se_end_final >= se_start_clamp:
                    se_range = { 'action_code': 'SE', 'start_frame': se_start_clamp, 'end_frame': se_end_final,
                                 'start_time': se_start_clamp / fps if fps > 0 else 0.0, 'end_time': se_end_time,
                                 'trigger_phrase': "(Inferred SE from O/E sequence)",
                                 'trigger_start': T_e_start if T_e_start is not None else (se_start_clamp / fps if fps > 0 else 0.0), 'trigger_end': trigger_end,
                                 'trigger_start_frame': se_start_frame, 'trigger_end_frame': trigger_end_frame_orig,
                                 'confidence_score': 101, 'index': range_index_counter, 'status': None, 'matched_words': matched_words,
                                 'original_event_start_time': original_event_start_time,
                                 'original_event_end_time': original_event_end_time
                                }
                    generated_action_ranges.append(se_range);
                    last_range_end_frame = se_end_final; range_index_counter += 1
                else: print(f"  Range SKIPPED (Inferred SE EndF {se_end_final} < StartF {se_start_clamp})")

            elif range_end_frame_final >= current_start_frame_for_range:
                 current_code = code if code != 'SO_SE' else 'SO'
                 if code == 'SO_SE': print(f"    Split Fallback: Generating normal 'SO' range for SO_SE event at F{event_start_frame}.")
                 range_end_time = range_end_frame_final / fps if fps > 0 else 0.0

                 extra_nm_data = {}
                 if current_code == 'NM':
                     extra_nm_data = {
                         'transcribed_text': current_event.get('transcribed_text', '?'),
                     }

                 new_range = {
                     'action_code': current_code,
                     'start_frame': current_start_frame_for_range,
                     'end_frame': range_end_frame_final,
                     'start_time': current_start_frame_for_range / fps if fps > 0 else 0.0, # Range time
                     'end_time': range_end_time, # Range time
                     'trigger_phrase': trigger_phrase,
                     'trigger_start': trigger_start, # Event trigger time
                     'trigger_end': trigger_end,     # Event trigger time
                     'trigger_start_frame': trigger_start_frame_orig,
                     'trigger_end_frame': trigger_end_frame_orig,
                     'confidence_score': confidence_score,
                     'index': range_index_counter,
                     'status': status,
                     'matched_words': matched_words,
                     'original_event_start_time': original_event_start_time,
                     'original_event_end_time': original_event_end_time,
                     **extra_nm_data # Add NM specific data
                 }
                 generated_action_ranges.append(new_range);
                 last_range_end_frame = range_end_frame_final; range_index_counter += 1
            else: print(f"  Range SKIPPED (Final EndF {range_end_frame_final} < Final StartF {current_start_frame_for_range}): Code='{code}'")

        # Post-Merge Logic (unchanged)
        merged_ranges = []
        if generated_action_ranges:
            try:
                generated_action_ranges.sort(key=lambda r: r.get('start_frame', 0))
                current_range = generated_action_ranges[0].copy()
                for i in range(1, len(generated_action_ranges)):
                    next_range = generated_action_ranges[i]
                    if next_range.get('action_code') == current_range.get('action_code') and \
                       next_range.get('status') == current_range.get('status') and \
                       next_range.get('start_frame') == current_range.get('end_frame', -1) + 1:
                        current_range['end_frame'] = next_range.get('end_frame'); current_range['end_time'] = next_range.get('end_time')
                    else:
                        merged_ranges.append(current_range); current_range = next_range.copy()
                merged_ranges.append(current_range)
            except IndexError: merged_ranges = []
            except Exception as e: print(f"  ERROR during post-merge: {e}"); merged_ranges = generated_action_ranges
        for idx, r in enumerate(merged_ranges): r['index'] = idx
        print(f"--- Generated {len(merged_ranges)} Final Action Ranges (after post-merge) ---")
        return merged_ranges

