#!/bin/bash

# Specify the name of the script you want to submit
SCRIPT_NAME="download_prepare_5yrs_ncl_slurm.sh"
echo "write the slurm script into ${SCRIPT_NAME}"
cat > ${SCRIPT_NAME} << 'EOF'
#!/bin/bash
#SBATCH -J generate_images_ncl_slurm       # Job name
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file`
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file`
#SBATCH -n 1               # Number of tasks
#SBATCH -c 4               # Number of CPUs per task (threads)
#SBATCH --mem=20G          # Memory per node (use units like G for gigabytes) - this job must need 20GB lol
#SBATCH -t 0-10:00         # Runtime in D-HH:MM format

module load imagemagick
if command -v convert >/dev/null 2>&1; then
    echo "Command exists."
else
    echo "Command does not exist."
fi

mkdir -p /groups/ESS3/zsun/data/AI_Emis/GLOB

echo "Loading NCL"
source /home/zsun/.bashrc
module load ncl
echo "Loaded NCL"

export YYYYMMDD="20160101"

echo "Drafting download_prepare_5yrs_training_data.ncl"
cat <<INNER_EOF >> download_prepare_5yrs_training_data.ncl

sdate=getenv("YYYYMMDD")
syyyy=str_get_cols(sdate,0,3)
iyyyy=stringtointeger(syyyy)
imm=stringtointeger(str_get_cols(sdate,4,5))
idd=stringtointeger(str_get_cols(sdate,6,7))
j0=greg2jul(iyyyy,1,1,-1)
jday=greg2jul(iyyyy,imm,idd,-1)
iww=(jday-j0)/7+1
sww=tostring_with_format(iww,"%0.3i")
dx=0.05

dir_out="/groups/ESS3/zsun/data/AI_Emis/GLOBv2"

;dir_out="."
fn_out=dir_out+"/firedata_"+sdate+".txt"

fn_grid="/groups/ESS3/yli74/data/S2S/emission_v4/ENS_EMISv4/2016/ENS_Emis_20161231.nc"
dir_frp="/groups/ESS3/yli74/data/S2S/emission_v4/ENS_EMISv4/"+syyyy
dir_fwi="/groups/ESS3/yli74/data/FWI/ORI/"+syyyy
dir_gdas="/scratch/yli74/data/FNL"
dir_igbp="/groups/ESS/yli74/data/IGBP/0.05_degree"
dir_vhi="/groups/ESS/yli74/data/VHI/"+syyyy
;dir_vhi="/groups/ESS3/sma8/data/GHG/ODIAC"

hlist=[/"LAT,LON,FRP,FWI,RH,T,U,V,RAIN,VHI_AVE,Land_Use,Nearest_1,Nearest_2,Nearest_3,Nearest_4,Nearest_5,Nearest_6,Nearest_7,Nearest_8,Nearest_9,Nearest_10,Nearest_11,Nearest_12,Nearest_13,Nearest_14,Nearest_15,Nearest_16,Nearest_17,Nearest_18,Nearest_19,Nearest_20,Nearest_21,Nearest_22,Nearest_23,Nearest_24"/]
write_table(fn_out,"w", hlist, "%s")

latmax=90
latmin=-90
lonmax=-180
lonmin=180

;1. read grid
f1=addfile(fn_grid,"r")
mlat=f1->lat  ;lat(lat)1791
mlon=f1->lon  ;lon(lon)3600
nx=dimsizes(mlon)
ny=dimsizes(mlat)
print("grid ready")
var=new((/nx*ny,35/),float)
var=-999.0
var@_FillValue=-999.0
vartemp=new((/ny,nx,35/),float)
vartemp=-999.0
vartemp@_FillValue=-999.0

;2.frp
f1=addfile(dir_frp+"/ENS_Emis_"+sdate+".nc","r")
frp=tofloat(f1->FRP_MMA2(0,:,:))
frp=where(frp.eq.frp@_FillValue,-999.0,frp)
frp@_FillValue=-999.0
delete(f1)
print("frp ready")

;3.fwi
ilon_fwi=asciiread("./mask/lon_index_fwi_glob.txt",-1,"integer")
ilat_fwi=asciiread("./mask/lat_index_fwi_glob.txt",-1,"integer")
f1=addfile(dir_fwi+"/ECMWF_FWI_FWI_"+sdate+"_1200_hr_v4.0_con.nc","r")
fwi=f1->fwi(0,:,:)
fwi=where(fwi.eq.fwi@_FillValue,-999.0,fwi)
fwi@_FillValue=-999.0
delete(f1)
print("fwi ready")

;4.gdas
ilon_gdas=asciiread("./mask/lon_index_gdas_glob.txt",-1,"integer")
ilat_gdas=asciiread("./mask/lat_index_gdas_glob.txt",-1,"integer")
f1=addfile(dir_gdas+"/gdas1.fnl0p25."+sdate+"00.f00.grib2","r")
f2=addfile(dir_gdas+"/gdas1.fnl0p25."+sdate+"06.f00.grib2","r")
f3=addfile(dir_gdas+"/gdas1.fnl0p25."+sdate+"12.f00.grib2","r")
f4=addfile(dir_gdas+"/gdas1.fnl0p25."+sdate+"18.f00.grib2","r")
temp=f1->TMP_P0_L1_GLL0
tnx=dimsizes(temp(0,:))
tny=dimsizes(temp(:,0))
print("gdas ready")

;t
ttemp=new((/4,tny,tnx/),float)
ttemp@_FillValue=temp@_FillValue
ttemp(0,:,:)=f1->TMP_P0_L1_GLL0
ttemp(1,:,:)=f2->TMP_P0_L1_GLL0
ttemp(2,:,:)=f3->TMP_P0_L1_GLL0
ttemp(3,:,:)=f4->TMP_P0_L1_GLL0
t=dim_avg_n(ttemp,0)
t=where(t.eq.temp@_FillValue,-999.0,t)
t@_FillValue=-999.0
delete([/temp,ttemp/])
print("t ready")

;rh
nh=25
temp=f1->RH_P0_L100_GLL0(nh,:,:)
ttemp=new((/4,tny,tnx/),float)
ttemp@_FillValue=temp@_FillValue
ttemp(0,:,:)=f1->RH_P0_L100_GLL0(nh,:,:)
ttemp(1,:,:)=f2->RH_P0_L100_GLL0(nh,:,:)
ttemp(2,:,:)=f3->RH_P0_L100_GLL0(nh,:,:)
ttemp(3,:,:)=f4->RH_P0_L100_GLL0(nh,:,:)
rh=dim_avg_n(ttemp,0)
rh=where(rh.eq.temp@_FillValue,-999.0,rh)
rh@_FillValue=-999.0
delete([/temp,ttemp/])
print("rh ready")

;u
temp=f1->UGRD_P0_L103_GLL0(0,:,:)
ttemp=new((/4,tny,tnx/),float)
ttemp@_FillValue=temp@_FillValue
ttemp(0,:,:)=f1->UGRD_P0_L103_GLL0(0,:,:)
ttemp(1,:,:)=f2->UGRD_P0_L103_GLL0(0,:,:)
ttemp(2,:,:)=f3->UGRD_P0_L103_GLL0(0,:,:)
ttemp(3,:,:)=f4->UGRD_P0_L103_GLL0(0,:,:)
u=dim_avg_n(ttemp,0)
u=where(u.eq.temp@_FillValue,-999.0,u)
u@_FillValue=-999.0
delete([/temp,ttemp/])
print("u ready")

;v
temp=f1->VGRD_P0_L103_GLL0(0,:,:)
ttemp=new((/4,tny,tnx/),float)
ttemp@_FillValue=temp@_FillValue
ttemp(0,:,:)=f1->VGRD_P0_L103_GLL0(0,:,:)
ttemp(1,:,:)=f2->VGRD_P0_L103_GLL0(0,:,:)
ttemp(2,:,:)=f3->VGRD_P0_L103_GLL0(0,:,:)
ttemp(3,:,:)=f4->VGRD_P0_L103_GLL0(0,:,:)
v=dim_avg_n(ttemp,0)
v=where(v.eq.temp@_FillValue,-999.0,v)
v@_FillValue=-999.0
delete([/temp,ttemp/])

;rain
temp=f1->PWAT_P0_L200_GLL0
ttemp=new((/4,tny,tnx/),float)
ttemp@_FillValue=temp@_FillValue
ttemp(0,:,:)=f1->PWAT_P0_L200_GLL0
ttemp(1,:,:)=f2->PWAT_P0_L200_GLL0
ttemp(2,:,:)=f3->PWAT_P0_L200_GLL0
ttemp(3,:,:)=f4->PWAT_P0_L200_GLL0
rain=dim_avg_n(ttemp,0)
rain=where(rain.eq.temp@_FillValue,-999.0,rain)
rain@_FillValue=-999.0
delete([/f1,temp,ttemp/])

;VHI
if (jday.eq.((iww-1)*7+j0)) then
  data=asciiread(dir_vhi+"/VHI."+syyyy+sww+".txt",-1,"string")
  tvhi=stringtofloat(str_get_field(data,3," "))
  tvhilat=stringtofloat(str_get_field(data,2," "))
  tvhilon=stringtofloat(str_get_field(data,1," "))
  tvhilon=where(tvhilon.ge.180,tvhilon-360,tvhilon)
  tlat=onedtond(tvhilat,(/3616,10000/))
  tlon=onedtond(tvhilon,(/3616,10000/))
  vhilat=tlat(:,0)
  vhilon=tlon(0,:)
  vhi=onedtond(tvhi,(/3616,10000/))
  delete([/data,tlat,tlon,tvhilon,tvhilat/])
else
  tvhi=asciiread("./temp/VHI."+syyyy+sww+".txt",-1,"float")
  vhi=onedtond(tvhi,(/ny,nx/))
  vartemp(:,:,9)=vhi
  delete([/tvhi,vhi/])
end if
;ilon_vhi=asciiread("./mask/lon_index_vhi_glob.txt",(/3600,10/),"integer")
;ilat_vhi=asciiread("./mask/lat_index_vhi_glob.txt",(/1791,10/),"integer")
;f1=addfile(dir_vhi+"/VHI."+syyyy+sww+".nc","r")
;vhilat=f1->lat
;vhilon=f1->lon
;vhilon=where(vhilon.ge.180,vhilon-360,vhilon)
;vhi=f1->VHI
;delete(f1)

;Land use
if (jday.eq.j0) then
  f1=addfile(dir_igbp+"/IGBP_"+syyyy+".nc","r")
  LUper=byte2flt(f1->Land_Cover_Type_1_Percent)
  LUper=where(LUper.ne.LUper@_FillValue,LUper,-999)
  lulat=tofloat(f1->latitude)
  lulon=tofloat(f1->longitude)
  lulon=where(lulon.ge.180,lulon-360,lulon)
  delete([/f1/])
else
  tLUper=asciiread("./temp/LU."+syyyy+".txt",-1,"float")
  LUper=onedtond(tLUper,(/ny,nx/))
  vartemp(:,:,10)=LUper
  delete([/tLUper,LUper/])
end if

;----add to var-----
do ix=2,nx-3
  do iy=2,ny-3
;1-2,lat lon
    vartemp(iy,ix,0)=(/mlat(iy)/)
    vartemp(iy,ix,1)=(/mlon(ix)/)
;3 frp
    vartemp(iy,ix,2)=(/frp(iy,ix)/)
;4 fwi
    vartemp(iy,ix,3)=(/fwi(ilat_fwi(iy),ilon_fwi(ix))/)
;5-8 t, rh, u, v
    vartemp(iy,ix,4)=(/t(ilat_gdas(iy),ilon_gdas(ix))/)
    vartemp(iy,ix,5)=(/rh(ilat_gdas(iy),ilon_gdas(ix))/)
    vartemp(iy,ix,6)=(/u(ilat_gdas(iy),ilon_gdas(ix))/)
    vartemp(iy,ix,7)=(/v(ilat_gdas(iy),ilon_gdas(ix))/)
    vartemp(iy,ix,8)=(/rain(ilat_gdas(iy),ilon_gdas(ix))/)
;9 VHI
    if (jday.eq.((iww-1)*7+j0)) then
      aa=ind((vhilat.ge.(mlat(iy)-dx)).and.(vhilat.le.(mlat(iy)+dx)))
      bb=ind((vhilon.ge.(mlon(ix)-dx)).and.(vhilon.le.(mlon(ix)+dx)))
      if ((.not.all(ismissing(aa))).and.(.not.all(ismissing(bb))))
        temp=ndtooned(vhi(aa,bb))
        if (.not.all(ismissing(temp)))
          vartemp(iy,ix,9)=avg(temp)
        end if
        delete(temp)
      end if
      delete([/aa,bb/])
    end if
;;10 lu
    if (jday.eq.j0) then
      temp=new(dimsizes(LUper(0,0,:)),float)
      aa=ind((lulat.ge.(mlat(iy)-dx)).and.(lulat.le.(mlat(iy)+dx)))
      bb=ind((lulon.ge.(mlon(ix)-dx)).and.(lulon.le.(mlon(ix)+dx)))
      if ((.not.all(ismissing(aa))).and.(.not.all(ismissing(bb)))) then
        do il=0,dimsizes(temp)-1
          temp(il)=avg(LUper(aa,bb,il))
        end do
        vartemp(iy,ix,10)=maxind(temp)
        delete(temp)
      end if
      delete([/aa,bb/])
    end if
;11- nearby FRP
    vartemp(iy,ix,11)=(/frp((iy),(ix+1))/)
    vartemp(iy,ix,12)=(/frp((iy-1),(ix+1))/)
    vartemp(iy,ix,13)=(/frp((iy-1),(ix))/)
    vartemp(iy,ix,14)=(/frp((iy-1),(ix-1))/)
    vartemp(iy,ix,15)=(/frp((iy),(ix-1))/)
    vartemp(iy,ix,16)=(/frp((iy+1),(ix-1))/)
    vartemp(iy,ix,17)=(/frp((iy+1),(ix))/)
    vartemp(iy,ix,18)=(/frp((iy+1),(ix+1))/)
    vartemp(iy,ix,19)=(/frp((iy),(ix+2))/)
    vartemp(iy,ix,20)=(/frp((iy-1),(ix+2))/)
    vartemp(iy,ix,21)=(/frp((iy-2),(ix+2))/)
    vartemp(iy,ix,22)=(/frp((iy-2),(ix+1))/)
    vartemp(iy,ix,23)=(/frp((iy-2),(ix))/)
    vartemp(iy,ix,24)=(/frp((iy-2),(ix-1))/)
    vartemp(iy,ix,25)=(/frp((iy-2),(ix-2))/)
    vartemp(iy,ix,26)=(/frp((iy-1),(ix-2))/)
    vartemp(iy,ix,27)=(/frp((iy),(ix-2))/)
    vartemp(iy,ix,28)=(/frp((iy+1),(ix-2))/)
    vartemp(iy,ix,29)=(/frp((iy+2),(ix-2))/)
    vartemp(iy,ix,30)=(/frp((iy+2),(ix-1))/)
    vartemp(iy,ix,31)=(/frp((iy+2),(ix))/)
    vartemp(iy,ix,32)=(/frp((iy+2),(ix+1))/)
    vartemp(iy,ix,33)=(/frp((iy+2),(ix+2))/)
    vartemp(iy,ix,34)=(/frp((iy+1),(ix+2))/)
  end do
end do
do iv=0,34
  var(:,iv)=ndtooned(vartemp(:,:,iv))
end do
write_table(fn_out,"a",\
         [/var(:,0),var(:,1),var(:,2),var(:,3),\
           var(:,4),var(:,5),var(:,6),var(:,7),\
           var(:,8),var(:,9),var(:,10),var(:,11),\
           var(:,12),var(:,13),var(:,14),var(:,15),\
           var(:,16),var(:,17),var(:,18),var(:,19),\
           var(:,20),var(:,21),var(:,22),var(:,23),\
           var(:,24),var(:,25),var(:,26),var(:,27),\
           var(:,28),var(:,29),var(:,30),var(:,31),\
           var(:,32),var(:,33),var(:,34)/],\
         "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f")
if (jday.eq.((iww-1)*7+j0)) then
  write_table("./temp/VHI."+syyyy+sww+".txt","w",[/var(:,9)/],"%f")
end if
if (jday.eq.j0) then
  write_table("./temp/LU."+syyyy+".txt","w",[/var(:,10)/],"%f")
end if

exit
INNER_EOF

echo "Start to run the NCL script: download_prepare_5yrs_training_data.ncl"

echo "ncl download_prepare_5yrs_training_data.ncl"

ncl download_prepare_5yrs_training_data.ncl

echo "Finished download_prepare_5yrs_training_data.ncl"

EOF

# Submit the Slurm job and wait for it to finish
echo "sbatch ${SCRIPT_NAME}"

# Submit the Slurm job
job_id=$(sbatch ${SCRIPT_NAME} | awk '{print $4}')
echo "job_id="${job_id}

if [ -z "${job_id}" ]; then
    echo "job id is empty. something wrong with the slurm job submission."
    exit 1
fi

# Wait for the Slurm job to finish
file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
previous_content=$(cat file_name)
exit_code=0
while true; do
    # Capture the current content
    file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
    current_content=$(<"${file_name}")

    # Compare current content with previous content
    diff_result=$(diff <(echo "$previous_content") <(echo "$current_content"))
    # Check if there is new content
    if [ -n "$diff_result" ]; then
        # Print the newly added content
        echo "$diff_result"
    fi
    # Update previous content
    previous_content="$current_content"


    job_status=$(scontrol show job ${job_id} | awk '/JobState=/{print $1}')
    #echo "job_status "$job_status
    #if [[ $job_status == "JobState=COMPLETED" ]]; then
    #    break
    #fi
    if [[ $job_status == *"COMPLETED"* ]]; then
        echo "Job $job_id has finished with state: $job_status"
        break;
    elif [[ $job_status == *"CANCELLED"* || $job_status == *"FAILED"* || $job_status == *"TIMEOUT"* || $job_status == *"NODE_FAIL"* || $job_status == *"PREEMPTED"* || $job_status == *"OUT_OF_MEMORY"* ]]; then
        echo "Job $job_id has finished with state: $job_status"
        exit_code=1
        break;
    fi
    sleep 10  # Adjust the sleep interval as needed
done

echo "Slurm job ($job_id) has finished."

echo "Print the job's output logs"
sacct --format=JobID,JobName,State,ExitCode,MaxRSS,Start,End -j $job_id
find /scratch/zsun/ -type f -name "*${job_id}.out" -exec cat {} \;
cat /scratch/zsun/test_data_slurm-*-$job_id.out

echo "All slurm job for ${SCRIPT_NAME} finishes."

exit $exit_code

