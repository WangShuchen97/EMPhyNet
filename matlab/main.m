clc;
clear;

datetime("now")

save_filename='./data/';
save_map_filename='./map_height/';
save_map_all_filename='./map_height_all/';
mapfilename='./map_data';

if ~exist(save_filename, 'dir')
    mkdir(save_filename);
end

if ~exist(save_map_filename, 'dir')
    mkdir(save_map_filename);
end

if ~exist(save_map_all_filename, 'dir')
    mkdir(save_map_all_filename);
end

t1=256;
t2=256;
tt1=256;
tt2=256; 
rtpm = propagationModel("raytracing", ...
    "Method","sbr", ...
    "AngularSeparation","high",...
    "MaxNumReflections",2, ...
    "MaxNumDiffractions",0,...
    "BuildingsMaterial","concrete", ...
    "SurfaceMaterial","concrete",...
    "TerrainMaterial","concrete");
rtPlusWeather = rtpm + propagationModel("gas");
delay=0:5e-9:3E-7;  

is_map_height=true;
is_data=true;
is_draw=false;
draw_num=7;
draw_rx=1000;

files = dir(fullfile(mapfilename));
size_row = size(files);
folder_num = size_row(1);

for num=1:1:folder_num
    if is_draw==true && num~=draw_num
        continue
    end
    
    fileName = files(num,1).name;
    if fileName=="."||fileName==".."
        continue
    end
    disp([num,folder_num]);
    disp(fileName);

    is_data_num=is_data && exist(strcat(save_filename,'data_',fileName(5:end),'.mat'))==0;
    is_map_height_num=is_map_height && exist(strcat(save_map_filename,'map_',fileName(5:end),'.mat'))==0;
    is_draw_num=is_draw;
    if is_data_num||is_map_height_num||is_draw_num
        try
        viewer = siteviewer("Buildings",[mapfilename,'/',fileName],"Basemap","topographic");%https://www.openstreetmap.org
        catch
        continue
        end
        
        tic
        
        fileName_split=strsplit(fileName,{'_',','});
        down=str2double(fileName_split{4});
        up=str2double(fileName_split{6});
        left=str2double(fileName_split{3});
        right=str2double(fileName_split{5});
                
        %=================================

        Latitude_all = linspace(down, up, t1);
        Longitude_all = linspace(left, right, t2);
        
        [Latitude_all, Longitude_all] = meshgrid(Latitude_all,Longitude_all);
        
        Latitude_all = Latitude_all(:)';
        Longitude_all = Longitude_all(:)';


        Coord=rfprop.internal.AntennaSiteCoordinates([Latitude_all(:), Longitude_all(:)],0,viewer);

        min_GroundHeight=min(Coord.GroundHeightAboveEllipsoid);
        max_GroundHeight=max(Coord.GroundHeightAboveEllipsoid);
        
        %=================================  

        %Tx
        Coord_tx=rfprop.internal.AntennaSiteCoordinates([(up+down)/2, (left+right)/2],2,viewer);

        tx = txsite("Name","Small cell transmitter", ...
        "Latitude",[(up+down)/2], ...
        "Longitude",[(left+right)/2], ...
        "AntennaHeight",[max(max_GroundHeight+1-Coord_tx.GroundHeightAboveEllipsoid,0)], ...
        "TransmitterPower",1, ...
        "TransmitterFrequency",4e8);

        if max_GroundHeight+1-Coord_tx.SurfaceHeightAboveEllipsoid<0
            close(viewer)
            continue
        end

        %=================================

        %Rx

        rx_height=max_GroundHeight+1-Coord.GroundHeightAboveEllipsoid;
        rx_height(rx_height < 0) = 0;
        rx_height=rx_height.';
        rx_num = rxsite("Name","Small cell receiver", ...
            "Latitude",Latitude_all, ...
            "Longitude",Longitude_all, ...
            "AntennaHeight",rx_height);
        rx_indicate_num=ones(1,t1*t2);
        for i=length(rx_num):-1:1
            if max_GroundHeight+1-Coord.SurfaceHeightAboveEllipsoid(i,1)<0
                rx_num(:,i)=[];
                rx_indicate_num(i)=0;
            end
        end
        kk=1;
        for i=1:t1*t2
            if rx_indicate_num(i)~=0
                rx_indicate_num(i)=kk;
                kk=kk+1;
            end
        end
        
        %=================================

        if is_map_height_num
            disp("map_height")

            Latitude_height = linspace(down, up, tt1);
            Longitude_height = linspace(left, right, tt2);
            
            [Latitude_height, Longitude_height] = meshgrid(Latitude_height,Longitude_height);
            
            Latitude_height = Latitude_height(:)';
            Longitude_height = Longitude_height(:)';

            Coord_height=rfprop.internal.AntennaSiteCoordinates([Latitude_height(:), Longitude_height(:)],0,viewer);

            map_all=[];
            map=[];
            k=1;
            index1=1;
            for i=down:(up-down)/(tt1-1):up
                index2=1;
                for j=left:(right-left)/(tt2-1):right
                    height=Coord_height.SurfaceHeightAboveEllipsoid(k,1);
                    if height<0.1
                        height=0;
                    end
                    map_all(index1,index2)=height;
                    if height>=max_GroundHeight+1
                        map(index1,index2)=1;
                    else
                        map(index1,index2)=0;
                    end
                    k=k+1;
                    index2=index2+1;
                end
                index1=index1+1;
            end
            save(strcat(save_map_all_filename,'map_',fileName(5:end),'.mat'),'map_all','-v7.3');
            save(strcat(save_map_filename,'map_',fileName(5:end),'.mat'),'map','-v7.3');
        end

        if is_data_num||is_draw_num
            
            %=================================

            if is_draw_num
                disp("draw")
                for i=1:t1*t2
                    if rx_indicate_num(1,i)==draw_rx
                        draw_rx_2=i;
                        break
                    end
                end
                show(tx)
                show(rx_num(1,draw_rx_2))
                gain=load(strcat(save_filename,'data_',fileName(5:end),'.mat'));
                gain=gain.gain;
                figure
                stem(delay,gain{1,draw_rx_2}, 'filled', 'y');
                
                figure
                plot(delay.',gain{1,5000}.','-','Color',[0.85, 0.33, 0.10],'LineWidth',2.5)
                hold on
                plot(delay.',gain{1,6000}.','-.','Color',[0.93, 0.69, 0.13],'LineWidth',2.5)
                hold on
                plot(delay.',gain{1,7000}.','--','Color',[0.00, 0.45, 0.74],'LineWidth',2.5)
                xlabel('Delay')
                ylabel('Channel Impulse Response')
                legend('({i_1}, {j_1})','({i_2}, {j_2})','({i_3}, {j_3})');
                
                figure
                set(gca,'FontSize',20);
                internal=5;
                start_num=6000;
                for i=1:20
                    j=i*internal;
                    show(rx_num(1,rx_indicate_num(1,start_num+j)))
                    if mod(i,3)==0
                        plot3(ones(size(delay,2),1)+i-1, delay.',gain{1,start_num+j}.','-','Color',[0.85, 0.33, 0.10],'LineWidth',2)
                        hold on
                    else
                        plot3(ones(size(delay,2),1)+i-1, delay.',gain{1,start_num+j}.','--','Color',[0.00, 0.45, 0.74],'LineWidth',1.5)
                        hold on
                    end
                end
                ylabel('Delay','fontsize',20)
                xlabel('Position','fontsize',20)
                zlabel('Channel Impulse Response','fontsize',20)
                grid on;
               
                continue
            end
            
            %=================================

            if is_data_num
                disp("data")
                rays=raytrace(tx,rx_num,rtpm,"Map", viewer);
                %==========Delete non planar rays========
                for kk=1:1:size(rays,2)
                    for t=size(rays{1,kk},2):-1:1
                        if rays{1,kk}(1,t).LineOfSight<1
                            temp_rays={rays{1,kk}(1,t).Interactions.Location};
                            for tt=1:1:size(temp_rays,2)
                                if abs(temp_rays{1,tt}(3)-(max_GroundHeight+1))>0.1
                                    rays{1,kk}(:,t)=[];
                                    break
                                end
                            end
                        else
                            rays{1,kk}(1,t).Interactions.Location=[0;0;0];
                        end
                    end
                end
                 %=================================
                
                rays_save={};
                k=1;
                kk=1;
                index1=1;
                for i=down:(up-down)/(t1-1):up
                    index2=1;
                    for j=left:(right-left)/(t2-1):right
                        temp=struct();
                        if rx_indicate_num(1,k)>0
                            if size(rays{1,kk},2)==0
                                rays_save{index1,index2}=jsonencode("NoRay");
                            else
                                temp.D=cell2mat({rays{1,kk}.PropagationDelay});
                                temp.L=cell2mat({rays{1,kk}.PathLoss});
                                temp.P=cell2mat({rays{1,kk}.PhaseShift});
                                temp.AOD=cell2mat({rays{1,kk}.AngleOfDeparture});
                                temp.AOD=mat2cell(temp.AOD, 2, ones(1,size(temp.AOD,2)));
                                temp.AOA=cell2mat({rays{1,kk}.AngleOfArrival});
                                temp.AOA=mat2cell(temp.AOA, 2, ones(1,size(temp.AOA,2)));

                                temp_rays={};
                                for t=1:1:size(rays{1,kk},2)
                                    
                                    temp2=rays{1,kk}(1,t).Interactions;
                                    temp2=arrayfun(@(x) x.Location, temp2, 'UniformOutput', false);
                                    temp2_n = numel(temp2);
                                    temp2_mat = zeros(temp2_n, 3);
                                    for temp2_n_i = 1:1:temp2_n
                                        temp2_mat(temp2_n_i, :) = temp2{temp2_n_i}'; 
                                    end
                                    temp_rays{1,t}=temp2_mat;
                                end

                                temp.ray=temp_rays;
                                
                                rays_save{index1,index2}=jsonencode(temp);
                            end
                            kk=kk+1;
                        else
                            rays_save{index1,index2}=jsonencode("NoRx");
                        end
                        
                        k=k+1;
                        index2=index2+1;
                    end
                    index1=index1+1;
                end
                save(strcat(save_filename,'data_',fileName(5:end),'.mat'),'delay','rays_save','max_GroundHeight','min_GroundHeight','-v7.3');
            end
        end
        toc
        close(viewer)
    end
end
datetime("now") 

