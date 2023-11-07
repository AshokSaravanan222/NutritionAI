import React, { useState, useEffect, useRef } from "react";
import { SafeAreaView, View, StyleSheet } from "react-native";
import { COLORS, SIZES, FONT } from "../../../constants";
import { FoodList, CameraButton } from "../../../components";
import { Stack } from "expo-router";
import TextTicker from "react-native-text-ticker";
import Slide from "../../../components/hero/Slide";

const Page = () => {
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: COLORS.lightWhite }}>
        <Stack.Screen
          options={{
            headerStyle: { backgroundColor: COLORS.secondary },
            headerShadowVisible: false,
            title: "",
          }}
        />
        <View
          style={{
            flex: 1,
            padding: SIZES.medium,
            paddingBottom: 90, // Add padding to create space for the button
          }}
            >
          <FoodList />
        </View>
        <View style={styles.cameraButtonContainer}>
          <CameraButton />
        </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  cameraButtonContainer: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    padding: SIZES.medium,
    backgroundColor: "transparent",
  },
  slide4: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
});

export default Page;