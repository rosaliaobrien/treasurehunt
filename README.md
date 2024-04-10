# TREASUREHUNT Pipeline to Align HST Exposures to Gaia

This pipeline was created to align Hubble flat-fielded exposures (FLC files) to the Gaia reference frame for the TREASUREHUNT program. Tutorials are available in two separate notebooks: *1) Drizzle Images* and *2) Align Drizzled Images to Gaia*. The first notebook initially stacks ("drizzles") the images together. The second notebook then uses the relatively high signal-to-noise drizzled image to calculate offsets from the Gaia reference frame. It updates the world coordinate system information within the individual exposures to align with Gaia. They are then re-drizzled, such that the final product is aligned to Gaia to within 10 mas.

The TREASUREHUNT program imaged the JWST North Ecliptic Pole Time Domain Field with the F275W, F435W and F606W filters. Please see [O'Brien+2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240104944O/abstract) for example use of this data and pipeline.

This notebook utilizes TweakReg and AstroDrizzle to perform the stacking of individual exposures. TweakReg has options to align to an externel reference frame (e.g., Gaia; please see Notebook tutorial [here](https://github.com/spacetelescope/notebooks/blob/master/notebooks/DrizzlePac/align_to_catalogs/align_to_catalogs.ipynb)). However, TweakReg can struggle when there are many cosmic rays and few stars in the FOV. This pipeline was created to assist with aligning to an external reference frame when TweakReg fails.

