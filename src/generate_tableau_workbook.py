"""
Tableau Workbook (.twb) Programmatic XML Generator
Generates an executable Tableau Workbook file (Customer_Churn_Dashboard.twb)
pointing directly to churn_predictions.csv in the current working directory.
"""

import os


def generate_tableau_twb(
    csv_filename="data/processed/churn_predictions_v2.csv",
    output_twb="dashboard/Customer_Churn_Dashboard.twb",
):
    if not os.path.exists(csv_filename):
        csv_filename = (
            "data/processed/churn_predictions.csv"
            if os.path.exists("data/processed/churn_predictions.csv")
            else "churn_predictions.csv"
        )

    os.makedirs(os.path.dirname(output_twb), exist_ok=True)
    abs_csv_path = os.path.abspath(csv_filename)
    csv_dir = os.path.dirname(abs_csv_path)
    base_csv_name = os.path.basename(abs_csv_path)

    xml_content = f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <preferences>
    <preference name='ui.encoding.shelf.height' value='24' />
    <preference name='ui.shelf.height' value='24' />
  </preferences>
  <datasources>
    <datasource caption='churn_predictions' inline='true' name='federated.churn_predictions' version='18.1'>
      <connection class='federated'>
        <named-connections>
          <named-connection caption='churn_predictions' name='textscan.churn_predictions'>
            <connection class='textscan' directory='{csv_dir}' filename='{base_csv_name}' password='' server='' />
          </named-connection>
        </named-connections>
        <relation connection='textscan.churn_predictions' name='{base_csv_name}' table='[{base_csv_name}]' type='table'>
          <columns character-set='UTF-8' header='yes' locale='en_US' separator=',' />
        </relation>
        <metadata-records>
          <metadata-record class='column'>
            <remote-name>customerID</remote-name>
            <remote-type>129</remote-type>
            <local-name>[customerID]</local-name>
            <parent-name>[{base_csv_name}]</parent-name>
            <datatype>string</datatype>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>Contract</remote-name>
            <remote-type>129</remote-type>
            <local-name>[Contract]</local-name>
            <parent-name>[{base_csv_name}]</parent-name>
            <datatype>string</datatype>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>PaymentMethod</remote-name>
            <remote-type>129</remote-type>
            <local-name>[PaymentMethod]</local-name>
            <parent-name>[{base_csv_name}]</parent-name>
            <datatype>string</datatype>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>InternetService</remote-name>
            <remote-type>129</remote-type>
            <local-name>[InternetService]</local-name>
            <parent-name>[{base_csv_name}]</parent-name>
            <datatype>string</datatype>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>tenure_group</remote-name>
            <remote-type>129</remote-type>
            <local-name>[tenure_group]</local-name>
            <parent-name>[{base_csv_name}]</parent-name>
            <datatype>string</datatype>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>tenure</remote-name>
            <remote-type>20</remote-type>
            <local-name>[tenure]</local-name>
            <parent-name>[{base_csv_name}]</parent-name>
            <datatype>integer</datatype>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>MonthlyCharges</remote-name>
            <remote-type>5</remote-type>
            <local-name>[MonthlyCharges]</local-name>
            <parent-name>[{base_csv_name}]</parent-name>
            <datatype>real</datatype>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>TotalCharges</remote-name>
            <remote-type>5</remote-type>
            <local-name>[TotalCharges]</local-name>
            <parent-name>[{base_csv_name}]</parent-name>
            <datatype>real</datatype>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>actual_churn</remote-name>
            <remote-type>20</remote-type>
            <local-name>[actual_churn]</local-name>
            <parent-name>[{base_csv_name}]</parent-name>
            <datatype>integer</datatype>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>predicted_churn</remote-name>
            <remote-type>20</remote-type>
            <local-name>[predicted_churn]</local-name>
            <parent-name>[{base_csv_name}]</parent-name>
            <datatype>integer</datatype>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>churn_probability</remote-name>
            <remote-type>5</remote-type>
            <local-name>[churn_probability]</local-name>
            <parent-name>[{base_csv_name}]</parent-name>
            <datatype>real</datatype>
          </metadata-record>
        </metadata-records>
      </connection>
      <aliases enabled='yes' />
      <column datatype='integer' name='[actual_churn]' role='measure' type='quantitative' />
      <column datatype='integer' name='[predicted_churn]' role='measure' type='quantitative' />
      <column datatype='real' name='[churn_probability]' role='measure' type='quantitative' />
      <column datatype='integer' name='[tenure]' role='measure' type='quantitative' />
      <column datatype='real' name='[MonthlyCharges]' role='measure' type='quantitative' />
      <column datatype='string' name='[customerID]' role='dimension' type='nominal' />
      <column datatype='string' name='[Contract]' role='dimension' type='nominal' />
      <column datatype='string' name='[PaymentMethod]' role='dimension' type='nominal' />
      <column datatype='string' name='[InternetService]' role='dimension' type='nominal' />
      <column datatype='string' name='[tenure_group]' role='dimension' type='nominal' />
      <column caption='Is High Risk' datatype='integer' name='[Calculation_HighRisk]' role='measure' type='quantitative'>
        <calculation class='tableau' formula='IF [churn_probability] &gt; 0.50 THEN 1 ELSE 0 END' />
      </column>
      <layout dim-ordering='alphabetic' dim-percentage='0.5' measure-ordering='alphabetic' measure-percentage='0.5' show-structure='true' />
    </datasource>
  </datasources>
  <worksheets>
    <worksheet name='KPI Banner'>
      <table>
        <view>
          <datasources>
            <datasource caption='churn_predictions' name='federated.churn_predictions' />
          </datasources>
          <datasource-dependencies datasource='federated.churn_predictions'>
            <column datatype='integer' name='[actual_churn]' role='measure' type='quantitative' />
            <column datatype='real' name='[MonthlyCharges]' role='measure' type='quantitative' />
            <column caption='Is High Risk' datatype='integer' name='[Calculation_HighRisk]' role='measure' type='quantitative' />
            <column-instance column='[actual_churn]' derivation='Avg' name='[avg:actual_churn:qk]' pivot='key' type='quantitative' />
            <column-instance column='[Calculation_HighRisk]' derivation='Sum' name='[sum:Calculation_HighRisk:qk]' pivot='key' type='quantitative' />
            <column-instance column='[MonthlyCharges]' derivation='Avg' name='[avg:MonthlyCharges:qk]' pivot='key' type='quantitative' />
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <style />
        <panes>
          <pane id='1'>
            <visualization version='18.1'>
              <pane-shapes>
                <shape type='text' />
              </pane-shapes>
            </visualization>
          </pane>
        </panes>
        <rows />
        <cols />
      </table>
    </worksheet>
    <worksheet name='Side-by-Side Bar Chart'>
      <table>
        <view>
          <datasources>
            <datasource caption='churn_predictions' name='federated.churn_predictions' />
          </datasources>
          <datasource-dependencies datasource='federated.churn_predictions'>
            <column datatype='string' name='[Contract]' role='dimension' type='nominal' />
            <column datatype='string' name='[tenure_group]' role='dimension' type='nominal' />
            <column datatype='real' name='[churn_probability]' role='measure' type='quantitative' />
            <column-instance column='[churn_probability]' derivation='Avg' name='[avg:churn_probability:qk]' pivot='key' type='quantitative' />
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <style />
        <panes>
          <pane id='1'>
            <visualization version='18.1'>
              <pane-shapes>
                <shape type='bar' />
              </pane-shapes>
            </visualization>
          </pane>
        </panes>
        <rows>[federated.churn_predictions].[avg:churn_probability:qk]</rows>
        <cols>([federated.churn_predictions].[Contract] / [federated.churn_predictions].[tenure_group])</cols>
      </table>
    </worksheet>
    <worksheet name='Risk Scatter Plot'>
      <table>
        <view>
          <datasources>
            <datasource caption='churn_predictions' name='federated.churn_predictions' />
          </datasources>
          <datasource-dependencies datasource='federated.churn_predictions'>
            <column datatype='string' name='[customerID]' role='dimension' type='nominal' />
            <column datatype='integer' name='[tenure]' role='measure' type='quantitative' />
            <column datatype='real' name='[MonthlyCharges]' role='measure' type='quantitative' />
            <column datatype='real' name='[churn_probability]' role='measure' type='quantitative' />
          </datasource-dependencies>
          <aggregation value='false' />
        </view>
        <style />
        <panes>
          <pane id='1'>
            <visualization version='18.1'>
              <pane-shapes>
                <shape type='circle' />
              </pane-shapes>
            </visualization>
          </pane>
        </panes>
        <rows>[federated.churn_predictions].[MonthlyCharges]</rows>
        <cols>[federated.churn_predictions].[tenure]</cols>
      </table>
    </worksheet>
    <worksheet name='Top 20 High-Risk Customers'>
      <table>
        <view>
          <datasources>
            <datasource caption='churn_predictions' name='federated.churn_predictions' />
          </datasources>
          <datasource-dependencies datasource='federated.churn_predictions'>
            <column datatype='string' name='[customerID]' role='dimension' type='nominal' />
            <column datatype='string' name='[Contract]' role='dimension' type='nominal' />
            <column datatype='real' name='[MonthlyCharges]' role='measure' type='quantitative' />
            <column datatype='real' name='[churn_probability]' role='measure' type='quantitative' />
          </datasource-dependencies>
        </view>
        <style />
        <panes>
          <pane id='1'>
            <visualization version='18.1'>
              <pane-shapes>
                <shape type='text' />
              </pane-shapes>
            </visualization>
          </pane>
        </panes>
        <rows>([federated.churn_predictions].[customerID] / ([federated.churn_predictions].[Contract] / ([federated.churn_predictions].[MonthlyCharges] / [federated.churn_predictions].[churn_probability])))</rows>
        <cols />
      </table>
    </worksheet>
  </worksheets>
  <dashboards>
    <dashboard name='Customer Churn Risk &amp; Retention Dashboard'>
      <style />
      <size maxheight='900' maxwidth='1400' minheight='900' minwidth='1400' />
      <zones>
        <zone h='100000' id='1' type='layout-basic' w='100000' x='0' y='0'>
          <zone h='10000' id='2' name='KPI Banner' type='sub-pane' w='100000' x='0' y='0' />
          <zone h='45000' id='3' name='Side-by-Side Bar Chart' type='sub-pane' w='50000' x='0' y='10000' />
          <zone h='45000' id='4' name='Risk Scatter Plot' type='sub-pane' w='50000' x='50000' y='10000' />
          <zone h='45000' id='5' name='Top 20 High-Risk Customers' type='sub-pane' w='100000' x='0' y='55000' />
        </zone>
      </zones>
    </dashboard>
  </dashboardsContainer>
</workbook>
"""

    with open(output_twb, "w", encoding="utf-8") as f:
        f.write(xml_content)

    print(f"[SUCCESS] Programmatic Tableau Workbook generated: '{output_twb}'")
    return output_twb


if __name__ == "__main__":
    generate_tableau_twb()
