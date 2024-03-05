import { SafeAreaView, View, StyleSheet } from "react-native";
import { COLORS, SIZES, FONT } from "../../../constants";
import { FoodList, CameraButton } from "../../../components";
import { Stack } from "expo-router";

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
  }
});

export default Page;