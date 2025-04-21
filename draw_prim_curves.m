% IP
default_file = 'IP_ori\exp-cplex-method-[''fixTop'']-perc-0.0-Mt3600-2024-01-18 16-23-54\primals.mat';
ori_fix_file = "IP_ori\exp-cplex-method-['fixTop']-perc-0.1-Mt600-2024-01-18 04-30-39\primals.mat";
ori_ps_file = "IP_ori\exp-cplex-method-['PS']-perc-0.2-Mt600-2024-01-18 05-50-44\primals.mat";
opt_fix_file = "IP_opt\exp-cplex-method-['fixTop']-perc-0.1-Mt600-2024-01-18 04-50-40\primals.mat";
opt_ps_file = "IP_opt\exp-cplex-method-['PS']-perc-0.1-Mt600-2024-01-18 11-06-54\primals.mat";
ori_ns_file = "IP_ori\exp-cplex-method-['node_selection', 'primal_mipstart']-perc-0.0-Mt600-2024-01-19 00-19-55\primals.mat";
opt_ns_file = "IP_opt\exp-cplex-method-['node_selection', 'primal_mipstart']-perc-0.0-Mt600-2024-01-19 00-39-55\primals.mat";

ylim([1.06,1.45]);
xlim([1,600]);



function draw_primal_curves(default_path,ori_fix_path,ori_ps_path,opt_fix_path,opt_ps_path,ori_ns_path,opt_ns_path)
    
    
    default_primals = load(default_path).primals;
    ori_fix = load(ori_fix_path).primals;
    ori_ps = load(ori_ps_path).primals;
    opt_fix = load(opt_fix_path).primals;
    opt_ps = load(opt_ps_path).primals;
    ori_ns = load(ori_ns_path).primals;
    opt_ns = load(opt_ns_path).primals;


    default_BKV = default_primals(:,size(default_primals,2));
    ori_fix_BKV = ori_fix(:,size(ori_fix,2));
    ori_ps_BKV = ori_ps(:,size(ori_ps,2));
    opt_fix_BKV = opt_fix(:,size(opt_fix,2));
    opt_ps_BKV = opt_ps(:,size(opt_ps,2));
    ori_ns_BKV = ori_ns(:,size(ori_ns,2));
    opt_ns_BKV = opt_ns(:,size(opt_ns,2));

    BKV= [default_BKV,ori_fix_BKV,ori_ps_BKV,opt_fix_BKV,opt_ps_BKV,ori_ns_BKV,opt_ns_BKV];
    BKV = min(BKV,[],2);
    
    pag_default = mean(default_primals./BKV,1);
    pag_ori_fix = mean(ori_fix./BKV,1);
    pag_ori_ps = mean(ori_ps./BKV,1);
    pag_opt_fix = mean(opt_fix./BKV,1);
    pag_opt_ps = mean(opt_ps./BKV,1);
    pag_ori_ns = mean(ori_ns./BKV,1);
    pag_opt_ns = mean(opt_ns./BKV,1);

    % statistic
    pag_default_std = std(default_primals./BKV,1);
    pag_ori_fix_std = std(ori_fix./BKV,1);
    pag_ori_ps_std = std(ori_ps./BKV,1);
    pag_opt_fix_std = std(opt_fix./BKV,1);
    pag_opt_ps_std = std(opt_ps./BKV,1);
    pag_ori_ns_std = std(ori_ns./BKV,1);
    pag_opt_ns_std = std(opt_ns./BKV,1);

    pag_default_mean = pag_default(600);
    pag_ori_fix_mean = pag_ori_fix(600);
    pag_ori_ps_mean = pag_ori_ps(600);
    pag_opt_fix_mean = pag_opt_fix(600);
    pag_opt_ps_mean = pag_opt_ps(600);
    pag_ori_ns_mean = pag_ori_ns(600);
    pag_opt_ns_mean = pag_opt_ns(600);

    pag_default_std = pag_default_std(600);
    pag_ori_fix_std = pag_ori_fix_std(600);
    pag_ori_ps_std = pag_ori_ps_std(600);
    pag_opt_fix_std = pag_opt_fix_std(600);
    pag_opt_ps_std = pag_opt_ps_std(600);
    pag_ori_ns_std = pag_ori_ns_std(600);
    pag_opt_ns_std = pag_opt_ns_std(600);

    
    
    red = [0.850980392156863 0.325490196078431 0.0980392156862745];
    green = [0.466666666666667 0.674509803921569 0.188235294117647];
    purple = [0.494117647058824 0.184313725490196 0.556862745098039];
    
    linwidth = 2;
    plot(1:size(pag_default,2),pag_default,'LineWidth',linwidth,'Marker','none','MarkerIndices',[1,100,200,300,400,500,600],'MarkerSize',8,'LineStyle','--');hold on;
    plot(1:size(pag_ori_fix,2),pag_ori_fix,'LineWidth',linwidth,'Marker','o','MarkerIndices',[1,100,200,300,400,500,600],'MarkerSize',8,'Color',red);hold on;
    plot(1:size(pag_opt_fix,2),pag_opt_fix,'LineWidth',linwidth,'Marker','diamond','MarkerIndices',[1,100,200,300,400,500,600],'MarkerSize',8,'Color',red);hold on;
    plot(1:size(pag_ori_ps,2),pag_ori_ps,'LineWidth',linwidth,'Marker','o','MarkerIndices',[1,100,200,300,400,500,600],'MarkerSize',8,'Color',green);hold on;
    plot(1:size(pag_opt_ps,2),pag_opt_ps,'LineWidth',linwidth,'Marker','diamond','MarkerIndices',[1,100,200,300,400,500,600],'MarkerSize',8,'Color',green);hold on;
    plot(1:size(pag_ori_ns,2),pag_ori_ns,'LineWidth',linwidth,'Marker','o','MarkerIndices',[1,100,200,300,400,500,600],'MarkerSize',8,'Color',purple);hold on;
    plot(1:size(pag_opt_ns,2),pag_opt_ns,'LineWidth',linwidth,'Marker','diamond','MarkerIndices',[1,100,200,300,400,500,600],'MarkerSize',8,'Color',purple);hold on;
    
    
    set(gca,'XTick',[100 200 300 400 500 600]);

    leg = legend('Default CPLEX','$r$-Fixing','$r_s$-Fixing','$r$-Local Branching','$r_s$-Local Branching','$r$-Node selection','$r_s$-Node selection');
    set(leg,'Interpreter','latex','FontSize',13,'Location','best','ItemTokenSize',[30,20]);
    grid on;
    xlabel('Solving time (s)','Interpreter','latex');
    ylabel('Primal gap','Interpreter','latex');
    
    

end



