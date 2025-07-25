LINEALPHA = 0.4
MARKERALPHA = 0.9
MARKERSIZE, LINEWIDTH = 3, 1
MARKERSIZEDIFF = 0

DissertationTheme = Theme(
    # Axis=(
    # xlabelsize=8,
    # ylabelsize=8,
    # titlesize=8
    # ),
    colgap=10,
    figure_padding=(1, 5, 1, 1),
    Axis=(
        titlesize=9,
        titlealign=:left,
        titlegap=1,
        titlefont="Times New Roman",
        xlabelfont="Times New Roman",
        ylabelfont="Times New Roman",
        xlabelsize=9,
        ylabelsize=9,
        xticklabelsize=7,
        yticklabelsize=7,
        xlabelpadding=0,
        ylabelpadding=0,
        topspinevisible=true,
        rightspinevisible=true,
        xtrimspine=false,
        ytrimspine=false,
        # topspinevisible=false,
        # rightspinevisible=false,
        # xtrimspine=true,
        # ytrimspine=true,
        xticklabelpad=0,
        yticklabelpad=2,
    ),
    Label=(;
        halign=:left,
        # tellwidth=false,
        #     # tellheight=false,
        justification=:right,
        # padding=(12, 0, 1, 0),
        #     # font="Times New Roman",
        fontsize=9,
    ),
    Lines=(
        linewidth=LINEWIDTH,
    ),
    ScatterLines=(
        ;
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        strokewidth=0.1,
    ),
    Scatter=(
        ;
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        # strokewidth=0.1,
    ),
    Legend=(;
        labelsize=9,
        labelfont="Times New Roman",
        # patchlabelgap=-8,
        patchsize=(10, 10),
        framevisible=false,
        rowgap=1,
    ),
    Colorbar=(
        ;
        spinewidth=0.5,
        tickwidth=0.5,
        ticksize=2,
    ),
)
