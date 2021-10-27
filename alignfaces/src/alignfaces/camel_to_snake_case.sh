old_name=(apertureTools contrastTools faceLandmarks makeAlignedFaces);
old_name+=(makeFiles phaseCong3 procrustesTools warpTools);
old_name+=(UnpackDictToVectorXy PackVectorXyAsDict PullJawlineToInsideOfHairline);
old_name+=(getRotationMatrix2D FilteredImagesCenteredOnCriticalBandAndWaveletBank);
old_name+=(PhaseCongruency FacialLandmarks);

new_name=(aperture_tools contrast_tools face_landmarks make_aligned_faces);
new_name+=(make_files phase_cong_3 procrustes_tools warp_tools);
new_name+=(unpack_dict_to_vector_xy pack_vector_xy_as_dict pull_jawline_to_inside_of_hairline);
new_name+=(get_rotation_matrix_2d filtered_images_centered_on_critical_band_and_wavelet_bank);
new_name+=(phase_congruency facial_landmarks);

all_modules=(*.py)  # get all module names
for file_i in "${all_modules[@]}"; do
    i=1;
    while [ $i -lt ${#old_name[*]} ]; do
        tempa=${old_name[$i]};
        tempb=${new_name[$i]};
        sed -i .orig "s/$tempa/$tempb/g" "$file_i";
        i=$(( $i + 1));
    done
done

# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
