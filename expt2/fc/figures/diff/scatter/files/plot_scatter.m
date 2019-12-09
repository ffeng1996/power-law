
layers = {'0', '1'};

ind = 1;
for i = 1:2
    figure(ind)
	current_data = importdata(char(strcat('mnist_2_1024_8_',layers(i),'.txt')));

    num_nodes = int32(size(current_data,1));
    x = current_data(1:num_nodes,1);
    y = current_data(1:num_nodes,2);
    z = current_data(1:num_nodes,4);
    scatter3(x, y, z, '.');
    set(gca,'xscale','log', 'yscale','log','zscale', 'log','FontSize',20);
    xlim([min(x) max(x)]);
    ylim([min(y) max(y)]);
    zlim([min(z), max(z)]);
    xlabel('$$d_1$$',  'Interpreter','latex','FontSize',20);
    ylabel('$$d_2$$', 'Interpreter','latex','FontSize',20);
    zlabel('$$\hat{\Delta}_0^l(d_1, d_2)$$', 'Interpreter','latex','FontSize',20);
    view(30, 10);
    box on;
    saveas(figure(ind), char(strcat('plots/scatter_3D/mnist_2_', ...
        num_units(i),'_',window_size(k),'_',layers(j),'.png')));
    ind = ind+1;
end
