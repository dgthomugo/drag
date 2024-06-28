using System;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(FMSocketIOManager))]
[CanEditMultipleObjects]
public class FMSocketIOManager_Editor : Editor
{
    private FMSocketIOManager FMSocketIO;

    private bool ShowSettings = false;

    SerializedProperty AutoInitProp;
    SerializedProperty NetworkTypeProp;


    SerializedProperty Settings_IPProp;
    SerializedProperty Settings_PortProp;
    SerializedProperty Settings_SslEnabledProp;
    SerializedProperty Settings_ReconnectDelayProp;

    SerializedProperty Settings_portRequiredProp;
    SerializedProperty Settings_socketIORequiredProp;

    SerializedProperty Settings_AckExpirationTimeProp;
    SerializedProperty Settings_PingIntervalProp;
    SerializedProperty Settings_PingTimeoutProp;
    SerializedProperty Settings_SocketIDProp;


    SerializedProperty ReadyProp;

    SerializedProperty OnReceivedByteDataEventProp;
    SerializedProperty OnReceivedStringDataEventProp;
    SerializedProperty OnReceivedRawMessageEventProp;

    SerializedProperty RawMessageRecordProp;

    void OnEnable()
    {

        AutoInitProp = serializedObject.FindProperty("AutoInit");
        NetworkTypeProp = serializedObject.FindProperty("NetworkType");

        Settings_IPProp = serializedObject.FindProperty("Settings.IP");
        Settings_PortProp = serializedObject.FindProperty("Settings.port");
        Settings_SslEnabledProp = serializedObject.FindProperty("Settings.sslEnabled");
        Settings_ReconnectDelayProp = serializedObject.FindProperty("Settings.reconnectDelay");

        Settings_portRequiredProp = serializedObject.FindProperty("Settings.portRequired");
        Settings_socketIORequiredProp = serializedObject.FindProperty("Settings.socketIORequired");

        Settings_AckExpirationTimeProp = serializedObject.FindProperty("Settings.ackExpirationTime");
        Settings_PingIntervalProp = serializedObject.FindProperty("Settings.pingInterval");
        Settings_PingTimeoutProp = serializedObject.FindProperty("Settings.pingTimeout");
        Settings_SocketIDProp = serializedObject.FindProperty("Settings.socketID");

        ReadyProp = serializedObject.FindProperty("Ready");

        OnReceivedByteDataEventProp = serializedObject.FindProperty("OnReceivedByteDataEvent");
        OnReceivedStringDataEventProp = serializedObject.FindProperty("OnReceivedStringDataEvent");
        OnReceivedRawMessageEventProp = serializedObject.FindProperty("OnReceivedRawMessageEvent");

        RawMessageRecordProp = serializedObject.FindProperty("RawMessageRecord");

    }

    // Update is called once per frame
    public override void OnInspectorGUI()
    {
        if (FMSocketIO == null) FMSocketIO = (FMSocketIOManager)target;

        serializedObject.Update();


        GUILayout.Space(10);
        GUILayout.BeginVertical("box");
        {
            GUILayout.Label("- Networking");
            GUILayout.BeginVertical("box");
            {
                GUILayout.BeginHorizontal();
                EditorGUILayout.PropertyField(AutoInitProp, new GUIContent("Auto Init"));
                GUILayout.EndHorizontal();
            }
            GUILayout.EndVertical();

            GUILayout.BeginVertical("box");
            {
                GUILayout.BeginHorizontal();
                EditorGUILayout.PropertyField(Settings_socketIORequiredProp, new GUIContent("Using SocketIO"));
                GUILayout.EndHorizontal();

                if (FMSocketIO.Settings.socketIORequired)
                {
                    GUILayout.BeginHorizontal();
                    EditorGUILayout.PropertyField(NetworkTypeProp, new GUIContent("NetworkType"));
                    GUILayout.EndHorizontal();
                }
            }
            GUILayout.EndVertical();

            GUILayout.BeginVertical("box");
            {
                GUILayout.BeginVertical();
                {
                    if (ShowSettings)
                    {
                        GUILayout.BeginHorizontal();
                        GUI.skin.button.alignment = TextAnchor.MiddleLeft;
                        if (GUILayout.Button("- Settings")) ShowSettings = false;
                        GUI.skin.button.alignment = TextAnchor.MiddleCenter;
                        GUILayout.EndHorizontal();

                        GUILayout.BeginVertical("box");
                        {
                            GUILayout.BeginHorizontal();
                            string _url = "ws" + (FMSocketIO.Settings.sslEnabled ? "s" : "") + "://" + FMSocketIO.Settings.IP;
                            if(FMSocketIO.Settings.portRequired) _url += ":" + FMSocketIO.Settings.port;
                            if (FMSocketIO.Settings.socketIORequired) _url += "/socket.io/...";
                            GUILayout.Label("URL: " + _url);
                            GUILayout.EndHorizontal();
                        }
                        GUILayout.EndVertical();

                        GUILayout.BeginVertical("box");
                        {
                            GUILayout.BeginHorizontal();
                            EditorGUILayout.PropertyField(Settings_IPProp, new GUIContent("IP"));
                            GUILayout.EndHorizontal();

                            GUILayout.BeginHorizontal();
                            EditorGUILayout.PropertyField(Settings_SslEnabledProp, new GUIContent("Ssl Enabled"));
                            GUILayout.EndHorizontal();
                        }
                        GUILayout.EndVertical();

                        GUILayout.BeginVertical("box");
                        {
                            GUILayout.BeginHorizontal();
                            EditorGUILayout.PropertyField(Settings_PortProp, new GUIContent("Port"));
                            GUILayout.EndHorizontal();

                            GUILayout.BeginHorizontal();
                            EditorGUILayout.PropertyField(Settings_portRequiredProp, new GUIContent("Port Required"));
                            GUILayout.EndHorizontal();
                        }
                        GUILayout.EndVertical();


                        GUILayout.BeginVertical("box");
                        {

                            GUILayout.BeginHorizontal();
                            EditorGUILayout.PropertyField(Settings_ReconnectDelayProp, new GUIContent("Reconnect Delay"));
                            GUILayout.EndHorizontal();

                            GUILayout.BeginHorizontal();
                            EditorGUILayout.PropertyField(Settings_AckExpirationTimeProp, new GUIContent("Ack Expiration Time"));
                            GUILayout.EndHorizontal();

                            GUILayout.BeginHorizontal();
                            EditorGUILayout.PropertyField(Settings_PingIntervalProp, new GUIContent("Ping Interval"));
                            GUILayout.EndHorizontal();

                            GUILayout.BeginHorizontal();
                            EditorGUILayout.PropertyField(Settings_PingTimeoutProp, new GUIContent("Ping Timeout"));
                            GUILayout.EndHorizontal();

                            GUILayout.BeginHorizontal();
                            EditorGUILayout.PropertyField(Settings_SocketIDProp, new GUIContent("Socket ID"));
                            GUILayout.EndHorizontal();
                        }
                        GUILayout.EndVertical();
                    }
                    else
                    {
                        GUILayout.BeginHorizontal();
                        GUI.skin.button.alignment = TextAnchor.MiddleLeft;
                        if (GUILayout.Button("+ Settings")) ShowSettings = true;
                        GUI.skin.button.alignment = TextAnchor.MiddleCenter;
                        GUILayout.EndHorizontal();
                    }
                }
                GUILayout.EndVertical();

                GUILayout.BeginHorizontal();
                EditorGUILayout.PropertyField(ReadyProp, new GUIContent("Ready"));
                GUILayout.EndHorizontal();
            }
            GUILayout.EndVertical();
        }
        GUILayout.EndVertical();

        GUILayout.Space(10);
        GUILayout.BeginVertical("box");
        {
            GUILayout.Label("- Receiver");
            GUILayout.BeginVertical("box");
            {
                GUILayout.BeginHorizontal();
                EditorGUILayout.PropertyField(OnReceivedByteDataEventProp, new GUIContent("OnReceivedByteDataEvent"));
                GUILayout.EndHorizontal();

                GUILayout.BeginHorizontal();
                EditorGUILayout.PropertyField(OnReceivedStringDataEventProp, new GUIContent("OnReceivedStringDataEvent"));
                GUILayout.EndHorizontal();
            }
            GUILayout.EndVertical();

            GUILayout.BeginVertical("box");
            {
                GUILayout.BeginHorizontal();
                EditorGUILayout.PropertyField(RawMessageRecordProp, new GUIContent("History: Raw Message (Max 20 lines)"));
                GUILayout.EndHorizontal();

                GUILayout.BeginHorizontal();
                EditorGUILayout.PropertyField(OnReceivedRawMessageEventProp, new GUIContent("OnReceivedRawMessageEvent"));
                GUILayout.EndHorizontal();
            }
            GUILayout.EndVertical();
        }
        GUILayout.EndVertical();


        //GUILayout.BeginVertical("box");
        //{
        //    GUILayout.Label("- Debug");
        //    GUILayout.BeginVertical("box");
        //    {
        //        GUILayout.BeginHorizontal();
        //        GUILayout.Label("Status: " + FMSocketIO.Status);
        //        GUILayout.EndHorizontal();
        //    }
        //    GUILayout.EndVertical();

        //    GUILayout.BeginVertical("box");
        //    {
        //        GUILayout.BeginHorizontal();
        //        EditorGUILayout.PropertyField(UIStatusProp, new GUIContent("UIStatus"));
        //        GUILayout.EndHorizontal();

        //        GUILayout.BeginHorizontal();
        //        EditorGUILayout.PropertyField(ShowLogProp, new GUIContent("ShowLog"));
        //        GUILayout.EndHorizontal();
        //    }
        //    GUILayout.EndVertical();
        //}
        //GUILayout.EndVertical();


        serializedObject.ApplyModifiedProperties();
    }
}