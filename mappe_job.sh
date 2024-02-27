#! /bin/bash
/var/services/homes/admin/miniconda3/envs/mappe/bin/python /var/services/homes/admin/mappe_rem/rem-map-main/mappe_rem.py -v temp_val -d $(date +%Y%m%d_%H%M) -i /volume1/web/dati_rete/ -o /var/services/homes/admin/mappe_rem/rem-map-main/maps/
