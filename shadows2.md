The shadow computation primarily takes place within the `illuminazione` function, which evaluates the terrain against the Sun's position, and the `frazione_disco` function, which handles partial solar occlusion.

* **Self-Shadowing (Ombra propria):** This is calculated as the dot product between the terrain's normal vectors and the Sun's directional unit vector.


* The Sun's unit vector in East-North-Up coordinates is computed using `sole = direzione_enu(az_deg, el_deg)`.


* The cosine of the incidence angle is evaluated with `coseno = np.clip(normali @ sole, 0.0, None)`, which forces negative values (surfaces pointing away from the Sun) to `0.0`.




* **Cast Shadows (Ombra portata):** The script ray-marches across the terrain in the Sun's direction to find the maximum elevation angle of any obstructing terrain, representing the local horizon.


* A loop iterates over progressive step distances: `for d in distanze:`.


* At each distance `d`, it samples the terrain elevation `zt` along the ray using `campiona_griglia(z_ext, passo_m, n_ext, e0 + d * s_e, n0 + d * s_n)`.


* It calculates the tangent of the obstruction angle and continuously updates the maximum found: `np.maximum(tan_max, (zt - z0) / d, out=tan_max)`.


* An optimization condition (`if el_deg - RAGGIO_SOLE_GRADI > 0 and (z_top - z0_min) / d < tan_limite: break`) stops the ray-march early if the highest point on the terrain (`z_top`) is too far away to block the Sun at the current elevation.


* The maximum tangent is converted to degrees to define the blocking horizon: `orizzonte = np.degrees(np.arctan(tan_max))`.




* **Partial Illumination (Illuminazione parziale):** Instead of treating shadows as strictly binary, the script calculates the exact geometric fraction of the solar disk that remains visible.


* The script passes the solar elevation and the terrain horizon to `frazione_disco(el_deg, orizzonte)`.


* Inside this function, `x = np.clip((el_deg - orizzonte_deg) / RAGGIO_SOLE_GRADI, -1.0, 1.0)` normalizes the position of the horizon line across the solar disk, which uses a predefined constant radius of 0.2666 degrees (`RAGGIO_SOLE_GRADI`).


* The visible area of the circular segment is computed using the exact area formula: `1.0 - (np.arccos(x) - x * np.sqrt(1.0 - x**2)) / np.pi`.





The final illumination intensity for each grid point is returned as the product of the exposed solar disk area and the incidence angle: `return frazione_disco(el_deg, orizzonte) * coseno`.