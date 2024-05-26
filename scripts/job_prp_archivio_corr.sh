#! /bin/bash
sleep 60
/var/services/homes/admin/miniconda3/envs/mappe/bin/python /var/services/homes/admin/mappe_rem/rem-map-main/mappe_rem_archivio.py -v rain_day -d $1_2350 -i /volume1/web/dati_rete/ -o /volume1/web/maps/archivio/
