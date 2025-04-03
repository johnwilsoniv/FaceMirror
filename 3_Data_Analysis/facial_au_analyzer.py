def cleanup_extracted_frames(self):
    """ Removes the _original.png AND the corresponding .jpg files after processing. """
    if not hasattr(self, 'frame_paths') or not self.frame_paths:
        logger.debug(f"({self.patient_id}) No frame paths stored, skipping cleanup.")
        return
    if not isinstance(self.frame_paths, dict):
        logger.warning(
            f"({self.patient_id}) frame_paths is not a dictionary ({type(self.frame_paths)}). Skipping cleanup.")
        return

    logger.info(f"({self.patient_id}) Cleaning up temporary extracted frame files (_original.png and .jpg)...")
    removed_png_count = 0
    removed_jpg_count = 0

    # Iterate through the paths stored in the dictionary (which are the _original.png paths)
    # Use list() to avoid issues if modifying dict during iteration (though we aren't here)
    for action, frame_path_png in list(self.frame_paths.items()):
        # 1. Clean up the _original.png file
        if isinstance(frame_path_png, str) and frame_path_png.endswith("_original.png"):
            if os.path.exists(frame_path_png):
                try:
                    os.remove(frame_path_png)
                    removed_png_count += 1
                    logger.debug(f"({self.patient_id}) Removed original frame: {os.path.basename(frame_path_png)}")
                except OSError as e:
                    logger.error(f"({self.patient_id}) Error removing frame file {frame_path_png}: {e}")
            else:
                logger.warning(f"({self.patient_id}) Original frame file not found for cleanup: {frame_path_png}")

            # 2. Construct and clean up the corresponding .jpg file
            frame_path_jpg = frame_path_png.replace("_original.png", ".jpg")  # Construct JPG path
            if os.path.exists(frame_path_jpg):
                try:
                    os.remove(frame_path_jpg)
                    removed_jpg_count += 1
                    logger.debug(f"({self.patient_id}) Removed labeled frame: {os.path.basename(frame_path_jpg)}")
                except OSError as e:
                    logger.error(f"({self.patient_id}) Error removing frame file {frame_path_jpg}: {e}")
            else:
                # This might happen if the JPG saving failed earlier, but PNG was saved
                logger.warning(f"({self.patient_id}) Labeled frame file not found for cleanup: {frame_path_jpg}")

        elif not isinstance(frame_path_png, str):
            logger.warning(
                f"({self.patient_id}) Invalid frame path type found for action '{action}': {type(frame_path_png)}")
        # else: path is string but not _original.png - ignore

    if removed_png_count > 0 or removed_jpg_count > 0:
        logger.info(
            f"({self.patient_id}) Removed {removed_png_count} original PNG image(s) and {removed_jpg_count} labeled JPG image(s).")
    else:
        logger.info(f"({self.patient_id}) No temporary frame files found or removed during cleanup.")

    # Clear the paths dictionary after attempting cleanup
    self.frame_paths = {}


# --- MODIFIED: create_au_visualization call ---
def create_au_visualization(self, au_values_left, au_values_right, norm_au_values_left, norm_au_values_right, action,
                            frame_num, patient_output_dir):
    """ Creates the per-action AU visualization plot by calling the visualizer. """
    if not self.visualizer: logger.error(f"({self.patient_id}) Visualizer not initialized."); return
    try:
        # Get the original PNG path (should have been saved by extract_frames)
        frame_path_for_action_png = self.frame_paths.get(action)  # This is the _original.png path
        if not frame_path_for_action_png or not os.path.exists(frame_path_for_action_png):
            # Try constructing the JPG path as a fallback (in case cleanup was run partially?)
            frame_path_jpg = os.path.join(patient_output_dir,
                                          f"{action}_{self.patient_id}_AUs_original.png".replace('_original.png',
                                                                                                 '.jpg'))  # Construct potential JPG path name pattern
            logger.warning(
                f"({self.patient_id}) Original frame PNG missing for action '{action}' ({frame_path_for_action_png}). Checking for JPG: {frame_path_jpg}")
            if os.path.exists(frame_path_jpg):
                frame_path_for_action = frame_path_jpg
            else:
                logger.error(
                    f"({self.patient_id}) Neither PNG nor JPG frame found for action '{action}'. Cannot create full AU visualization.")
                return  # Skip if frame is essential
        else:
            frame_path_for_action = frame_path_for_action_png  # Use the PNG path if it exists

        # Call visualizer, passing self as first positional argument
        self.visualizer.create_au_visualization(
            self,  # Pass the analyzer instance (self)
            au_values_left,
            au_values_right,
            norm_au_values_left,
            norm_au_values_right,
            action,
            frame_num,
            patient_output_dir,
            # Pass the specific frame path needed by the visualizer
            frame_path=frame_path_for_action,  # Use the determined path (PNG or JPG fallback)
            action_descriptions=self.action_descriptions,
            action_to_aus=self.action_to_aus,
            results=self.results
        )
    except Exception as e:
        logger.error(
            f"({self.patient_id}) Error during visualizer.create_au_visualization call for action '{action}': {e}",
            exc_info=True)


# --- END MODIFICATION ---

# --- MODIFIED: create_symmetry_visualization call ---
def create_symmetry_visualization(self, patient_output_dir):
    """ Creates the symmetry plot by calling the visualizer. """
    if not self.visualizer: logger.error(f"({self.patient_id}) Visualizer not initialized."); return None
    if not self.results: logger.error(f"({self.patient_id}) No results for symmetry plot."); return None
    try:
        # Call visualizer, passing self as first positional argument
        output_path = self.visualizer.create_symmetry_visualization(
            self,  # Pass the analyzer instance (self)
            patient_output_dir,
            self.patient_id,
            self.results,
            self.action_descriptions
        )
        return output_path
    except Exception as e:
        logger.error(f"({self.patient_id}) Error during visualizer.create_symmetry_visualization call: {e}",
                     exc_info=True)
        return None


# --- END MODIFICATION ---

# --- MODIFIED: create_patient_dashboard call ---
def create_patient_dashboard(self):
    """ Creates an HTML dashboard by calling the visualizer. """
    if not self.visualizer: logger.error(f"({self.patient_id}) Visualizer not initialized."); return None
    if not self.results: logger.error(f"({self.patient_id}) No results for dashboard."); return None
    # Use the patient-specific output directory stored in the analyzer instance
    # Ensure self.output_dir was set correctly during frame extraction or manually
    patient_specific_dir = self.output_dir
    if not patient_specific_dir or not os.path.isdir(patient_specific_dir):
        # Attempt to reconstruct the path if needed (less ideal)
        if self.patient_id and os.path.isdir(os.path.join("../3.5_Results", self.patient_id)):
            patient_specific_dir = os.path.join("../3.5_Results", self.patient_id)
            logger.warning(f"({self.patient_id}) Reconstructed patient output directory: {patient_specific_dir}")
        else:
            logger.error(
                f"({self.patient_id}) Patient-specific output directory invalid or not set. Cannot create dashboard.");
            return None

    try:
        # Call visualizer with arguments matching its definition:
        # def create_patient_dashboard(self, analyzer, patient_output_dir, patient_id, results, action_descriptions):
        output_path = self.visualizer.create_patient_dashboard(
            analyzer=self,  # Pass the analyzer instance (self)
            patient_output_dir=patient_specific_dir,  # Pass the specific patient directory path
            patient_id=self.patient_id,  # Pass the patient ID
            results=self.results,  # Pass the results
            action_descriptions=self.action_descriptions  # Pass the descriptions
        )
        # Log the returned path (which should now be correct)
        if output_path:
            logger.info(f"({self.patient_id}) Dashboard created: {output_path}")
        else:
            logger.warning(f"({self.patient_id}) Dashboard creation returned None.")
        return output_path
    except Exception as e:
        logger.error(f"({self.patient_id}) Dashboard creation failed: {e}", exc_info=True)
        return None
# --- END MODIFICATION ---