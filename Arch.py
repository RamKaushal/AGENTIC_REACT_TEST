from graphviz import Digraph

dot = Digraph(comment='LangGraph Forecasting Pipeline', format='png')
dot.attr(rankdir='TB', layout='dot', fontsize='12', dpi='300')

# Global graph styling
dot.attr('graph', bgcolor='white', pad='0.5', nodesep='0.8', ranksep='1.2')
dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
dot.attr('edge', color='#333333', penwidth='2', arrowsize='0.8')

# Simplified approach - using HTML-like labels with clearer nesting
def add_node(name, label, tools):
    tool_rows = ""
    for tool in tools:
        tool_rows += f"<TR><TD ALIGN='LEFT' PORT='{tool}'>{tool}</TD></TR>"
    
    content = f"""<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8" BGCOLOR="#fef6ea">
        <TR><TD COLSPAN="1"><FONT POINT-SIZE="16"><B>{label}</B></FONT></TD></TR>
        <TR><TD>
            <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="2" CELLPADDING="6" BGCOLOR="#e3f2fd">
                <TR><TD BGCOLOR="#b3d9ff"><FONT POINT-SIZE="12"><I><B>React Agent</B></I></FONT></TD></TR>
                {tool_rows}
            </TABLE>
        </TD></TR>
    </TABLE>
    >"""
    
    dot.node(name, content)

# Create nodes
add_node('Corecast', 'Corecast', [
    'preprocess',
    'MAPE_Comparison_tool',
    'forecast_lstm_tool', 
    'forecast_xgb_tool'
])

add_node('FestivCast', 'FestivCast', [
    'generate_holidays_dataframe',
    'identify_holidays',
    'holiday_impact'
])

add_node('SesoCast', 'SesoCast', [
    'get_historical_periods_dynamic_actual',
    'get_historical_periods_dynamic_fcst'
])

add_node('PulseCast', 'PulseCast', [
    'Tool G',
    'Tool H'
])

add_node('Foresight', 'Foresight', [
    'Tool I',
    'Tool J'
])

# Create edges
dot.edge('Corecast', 'FestivCast')
dot.edge('Corecast', 'SesoCast') 
dot.edge('Corecast', 'PulseCast')

dot.edge('FestivCast', 'Foresight')
dot.edge('SesoCast', 'Foresight')
dot.edge('PulseCast', 'Foresight')

# Render
dot.render('langgraph_pipeline_fixed', view=True, cleanup=True)
