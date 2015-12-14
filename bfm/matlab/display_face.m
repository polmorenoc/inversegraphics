function display_face (shp, tex, tl, rp)
	
	shp = reshape(shp, [ 3 prod(size(shp))/3 ])'; 
	tex = reshape(tex, [ 3 prod(size(tex))/3 ])'; 
	tex = min(tex, 255);


	set(gcf, 'Renderer', 'opengl');
	fig_pos = get(gcf, 'Position');
	fig_pos(3) = rp.width;
	fig_pos(4) = rp.height;
	set(gcf, 'Position', fig_pos);
	set(gcf, 'ResizeFcn', @resizeCallback);

	mesh_h = trimesh(...
		tl, shp(:, 1), shp(:, 3), shp(:, 2), ...
		'EdgeColor', 'none', ...
		'FaceVertexCData', tex/255, 'FaceColor', 'interp', ...
		'FaceLighting', 'phong' ...
	);

	set(gca, ...
		'DataAspectRatio', [ 1 1 1 ], ...
		'PlotBoxAspectRatio', [ 1 1 1 ], ...
		'Units', 'pixels', ...
		'GridLineStyle', 'none', ...
		'Position', [ 0 0 fig_pos(3) fig_pos(4) ], ...
		'Visible', 'off', 'box', 'off', ...
		'Projection', 'perspective' ...
		); 
	
	set(gcf, 'Color', [ 0 0 0 ]); 
	view(180 + rp.phi * 180 / pi, 0);

	material([.5, .5, .1 1  ])
	camlight('headlight');

	
%% ------------------------------------------------------------CALLBACK--------
function resizeCallback (obj, eventdata)
	
	fig = gcbf;
	fig_pos = get(fig, 'Position');

	axis = findobj(get(fig, 'Children'), 'Tag', 'Axis.Head');
	set(axis, 'Position', [ 0 0 fig_pos(3) fig_pos(4) ]);
	
