#! /bin/bash
sleep 140
/var/services/homes/admin/miniconda3/envs/mappe/bin/python /var/services/homes/admin/mappe_rem/rem-map-main/mappe_rem.py -v wind_gust -d $(date +%Y%m%d_%H%M) -i /volume1/web/dati_rete/ -o /volume1/web/maps/
